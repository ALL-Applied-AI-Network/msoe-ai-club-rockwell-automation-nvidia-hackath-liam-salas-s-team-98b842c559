"""Author time-sampled, uncalibrated liquid appearance overrides.

This module is intentionally separate from :mod:`fluidmix.twin_appearance` so
the reviewed ``direct_appearance_v1`` contract remains unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import secrets

from fluidmix.twin_appearance import (
    MAX_APPEARANCE_LAYER_BYTES,
    MATERIAL_PATH,
    SOURCE_STAGE,
    _EXPECTED_LAYER_INFO_KEYS,
    _EXPECTED_PRIM_PATHS,
    _SHA256,
    _all_prim_specs,
    _canonical_relative_sublayer,
    _existing_directory,
    _float32,
    _fsync_directory,
    _import_pxr,
    _lexical_absolute,
    _regular_snapshot,
    _reject_symlink_components,
    _require_owned_identity,
    _source_record,
    _validate_source_stage,
    _write_exclusive,
    validate_appearance_request,
)


STAGE_APPEARANCE_SCHEMA = "direct_stage_appearance_v1"
_COMMAND_PLAN_FIELDS = (
    "schema",
    "variant_id",
    "stage_sha256",
    "stage_bytes",
    "source_stage",
    "source_stage_sha256",
    "material_path",
    "samples",
)
_CUSTOM_LAYER_FIELDS = frozenset(
    {
        "fluidmix_schema",
        "fluidmix_variant_id",
        "fluidmix_material_path",
        "fluidmix_stage_time_codes",
        "fluidmix_stage_diffuse_colors_linear",
        "fluidmix_stage_opacities",
        "fluidmix_source_stage",
        "fluidmix_source_stage_sha256",
        "fluidmix_uncalibrated",
    }
)
_EXPECTED_TIME_SAMPLED_ATTRIBUTE_INFO_KEYS = frozenset(
    {"typeName", "custom", "variability", "timeSamples"}
)


def validate_stage_appearance_request(
    variant_id: object,
    samples: object,
    *,
    start_time_code: int | float | None = None,
    end_time_code: int | float | None = None,
) -> dict[str, object]:
    """Validate one time-sampled appearance request without coercion."""

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("stage appearance samples must be a sequence")
    if not samples:
        raise ValueError("stage appearance samples must not be empty")
    for role, bound in (
        ("start time code", start_time_code),
        ("end time code", end_time_code),
    ):
        if bound is not None and (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(bound)
        ):
            raise TypeError(f"{role} must be a finite non-Boolean number")
    if (
        start_time_code is not None
        and end_time_code is not None
        and start_time_code > end_time_code
    ):
        raise ValueError("source time-code bounds are reversed")

    validated: list[dict[str, object]] = []
    previous: int | None = None
    expected_fields = {"time_code", "diffuse_color_linear", "opacity"}
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != expected_fields:
            raise ValueError(
                "each stage appearance sample must contain exactly time_code, "
                "diffuse_color_linear, and opacity"
            )
        time_code = sample["time_code"]
        if isinstance(time_code, bool) or not isinstance(time_code, int):
            raise TypeError(f"sample {index} time code must be a non-Boolean integer")
        if previous is not None and time_code <= previous:
            raise ValueError("stage appearance time codes must be strictly increasing")
        if start_time_code is not None and time_code < start_time_code:
            raise ValueError("stage appearance time code is before the source stage")
        if end_time_code is not None and time_code > end_time_code:
            raise ValueError("stage appearance time code is after the source stage")
        appearance = validate_appearance_request(
            variant_id,
            sample["diffuse_color_linear"],
            sample["opacity"],
        )
        validated.append(
            {
                "time_code": time_code,
                "diffuse_color_linear": list(
                    appearance["diffuse_color_linear"]
                ),
                "opacity": appearance["opacity"],
            }
        )
        previous = time_code
    return {"variant_id": variant_id, "samples": validated}


def stage_appearance_command_plan(inspection: Mapping[str, object]) -> dict[str, object]:
    """Project a strict inspection record into a preregistration object."""

    if not isinstance(inspection, Mapping):
        raise TypeError("stage appearance inspection must be a mapping")
    try:
        projected = {field: inspection[field] for field in _COMMAND_PLAN_FIELDS}
    except KeyError as error:
        raise ValueError(
            f"stage appearance inspection is missing {error.args[0]!r}"
        ) from error
    if (
        type(projected["schema"]) is not str
        or projected["schema"] != STAGE_APPEARANCE_SCHEMA
    ):
        raise ValueError("stage appearance inspection schema is invalid")
    request = validate_stage_appearance_request(
        projected["variant_id"], projected["samples"]
    )
    for field in ("stage_sha256", "source_stage_sha256"):
        value = projected[field]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"stage appearance inspection {field} is not a SHA-256")
    stage_bytes = projected["stage_bytes"]
    if (
        isinstance(stage_bytes, bool)
        or not isinstance(stage_bytes, int)
        or stage_bytes <= 0
        or stage_bytes > MAX_APPEARANCE_LAYER_BYTES
    ):
        raise ValueError("stage appearance inspection stage_bytes is outside the size bound")
    if type(projected["source_stage"]) is not str or projected["source_stage"] != SOURCE_STAGE:
        raise ValueError("stage appearance inspection source stage is invalid")
    if type(projected["material_path"]) is not str or projected["material_path"] != MATERIAL_PATH:
        raise ValueError("stage appearance inspection material path is invalid")
    projected["samples"] = request["samples"]
    return projected


def inspect_stage_appearance_layer(
    layer_path: str | os.PathLike[str],
    twin_root: str | os.PathLike[str],
    *,
    expected_variant_id: str | None = None,
    require_filename_match: bool = True,
) -> dict[str, object]:
    """Reopen and strictly inspect one time-sampled appearance root layer."""

    Gf, Sdf, Usd, Vt = _import_pxr()
    lexical_layer = _lexical_absolute(layer_path)
    if lexical_layer.suffix != ".usda":
        raise ValueError("stage appearance layer filename must end with .usda")
    layer_snapshot = _regular_snapshot(
        lexical_layer,
        "stage appearance layer",
        max_bytes=MAX_APPEARANCE_LAYER_BYTES,
    )
    twin, source_snapshot = _source_record(twin_root)
    source = Path(source_snapshot["path"])
    _, source_timing = _validate_source_stage(source, source_snapshot, Sdf, Usd)

    layer = Sdf.Layer.FindOrOpen(str(lexical_layer))
    if layer is None or not layer.Reload(force=True):
        raise ValueError("stage appearance layer is not a readable USD layer")
    if set(layer.pseudoRoot.ListInfoKeys()) != _EXPECTED_LAYER_INFO_KEYS:
        raise ValueError("stage appearance layer contains missing or extra root metadata")
    custom = dict(layer.customLayerData)
    if set(custom) != _CUSTOM_LAYER_FIELDS:
        raise ValueError("stage appearance custom metadata fields are missing or extra")
    if type(custom["fluidmix_schema"]) is not str or custom["fluidmix_schema"] != STAGE_APPEARANCE_SCHEMA:
        raise ValueError("stage appearance layer schema metadata is invalid")
    variant_id = custom["fluidmix_variant_id"]
    time_codes_raw = custom["fluidmix_stage_time_codes"]
    colors_raw = custom["fluidmix_stage_diffuse_colors_linear"]
    opacities_raw = custom["fluidmix_stage_opacities"]
    if not isinstance(time_codes_raw, Vt.IntArray):
        raise ValueError("stage appearance time-code metadata must be an int array")
    if not isinstance(colors_raw, Vt.DoubleArray):
        raise ValueError("stage appearance RGB metadata must be a double array")
    if not isinstance(opacities_raw, Vt.DoubleArray):
        raise ValueError("stage appearance opacity metadata must be a double array")
    if len(colors_raw) != 3 * len(time_codes_raw) or len(opacities_raw) != len(time_codes_raw):
        raise ValueError("stage appearance metadata arrays have inconsistent lengths")
    raw_samples = []
    for index, time_code in enumerate(time_codes_raw):
        raw_samples.append(
            {
                "time_code": int(time_code),
                "diffuse_color_linear": [
                    float(value) for value in colors_raw[3 * index : 3 * index + 3]
                ],
                "opacity": float(opacities_raw[index]),
            }
        )
    request = validate_stage_appearance_request(
        variant_id,
        raw_samples,
        start_time_code=source_timing["start_time_code"],
        end_time_code=source_timing["end_time_code"],
    )
    if expected_variant_id is not None and variant_id != expected_variant_id:
        raise ValueError("stage appearance variant metadata does not match expected variant")
    if require_filename_match and lexical_layer.stem != variant_id:
        raise ValueError("stage appearance filename stem does not match variant metadata")
    if type(custom["fluidmix_material_path"]) is not str or custom["fluidmix_material_path"] != MATERIAL_PATH:
        raise ValueError("stage appearance material-path metadata is invalid")
    if type(custom["fluidmix_source_stage"]) is not str or custom["fluidmix_source_stage"] != SOURCE_STAGE:
        raise ValueError("stage appearance source-stage metadata is invalid")
    if (
        type(custom["fluidmix_source_stage_sha256"]) is not str
        or custom["fluidmix_source_stage_sha256"] != source_snapshot["sha256"]
    ):
        raise ValueError("stage appearance source-stage hash does not match source bytes")
    if custom["fluidmix_uncalibrated"] is not True:
        raise ValueError("stage appearance layer must declare uncalibrated=true")

    if len(layer.subLayerPaths) != 1:
        raise ValueError("stage appearance layer must contain exactly one sublayer")
    if len(layer.subLayerOffsets) != 1 or layer.subLayerOffsets[0] != Sdf.LayerOffset():
        raise ValueError("stage appearance sublayer must use one identity layer offset")
    sublayer = layer.subLayerPaths[0]
    if (
        not isinstance(sublayer, str)
        or not sublayer
        or "\\" in sublayer
        or PurePosixPath(sublayer).is_absolute()
        or posixpath.normpath(sublayer) != sublayer
    ):
        raise ValueError("stage appearance sublayer must be one canonical relative path")
    if sublayer != _canonical_relative_sublayer(lexical_layer, source):
        raise ValueError("stage appearance sublayer does not resolve canonically to source")
    resolved = (lexical_layer.parent / Path(*PurePosixPath(sublayer).parts)).resolve(strict=True)
    if resolved != source:
        raise ValueError("stage appearance sublayer does not resolve to exact source stage")
    if layer.defaultPrim != "FillStation":
        raise ValueError("stage appearance default prim must be FillStation")
    layer_timing = {
        "start_time_code": float(layer.startTimeCode),
        "end_time_code": float(layer.endTimeCode),
        "frames_per_second": float(layer.framesPerSecond),
        "time_codes_per_second": float(layer.timeCodesPerSecond),
    }
    if layer_timing != source_timing:
        raise ValueError("stage appearance timing does not exactly match source stage")

    prims = _all_prim_specs(layer)
    if tuple(prims) != _EXPECTED_PRIM_PATHS:
        raise ValueError("stage appearance prim structure is missing, extra, or reordered")
    for path, prim in prims.items():
        if prim.specifier != Sdf.SpecifierOver or prim.typeName:
            raise ValueError("stage appearance may contain only untyped ancestor over specs")
        if set(prim.ListInfoKeys()) != {"specifier"}:
            raise ValueError(f"stage appearance prim {path} has unsupported metadata")
        expected_properties = (
            {"inputs:diffuseColor", "inputs:opacity"} if path == MATERIAL_PATH else set()
        )
        if {prop.name for prop in prim.properties} != expected_properties:
            raise ValueError("stage appearance layer has missing or extra shader properties")
    shader = prims[MATERIAL_PATH]
    diffuse = shader.attributes["inputs:diffuseColor"]
    opacity_attribute = shader.attributes["inputs:opacity"]
    if diffuse.typeName != Sdf.ValueTypeNames.Color3f:
        raise ValueError("stage appearance diffuseColor must have type color3f")
    if opacity_attribute.typeName != Sdf.ValueTypeNames.Float:
        raise ValueError("stage appearance opacity must have type float")
    expected_times = [float(sample["time_code"]) for sample in request["samples"]]
    for attribute in (diffuse, opacity_attribute):
        if (
            set(attribute.ListInfoKeys()) != _EXPECTED_TIME_SAMPLED_ATTRIBUTE_INFO_KEYS
            or attribute.custom
            or attribute.variability != Sdf.VariabilityVarying
            or attribute.default is not None
        ):
            raise ValueError("stage appearance shader property has unsupported metadata")
        if attribute.connectionPathList.GetAddedOrExplicitItems():
            raise ValueError("stage appearance shader properties must not contain connections")
        if layer.ListTimeSamplesForPath(attribute.path) != expected_times:
            raise ValueError("stage appearance shader properties have missing or extra samples")
    for sample in request["samples"]:
        time_code = float(sample["time_code"])
        expected_rgb = Gf.Vec3f(*[float(value) for value in sample["diffuse_color_linear"]])
        if layer.QueryTimeSample(diffuse.path, time_code) != expected_rgb:
            raise ValueError("stage appearance diffuseColor sample does not match metadata")
        if layer.QueryTimeSample(opacity_attribute.path, time_code) != _float32(sample["opacity"]):
            raise ValueError("stage appearance opacity sample does not match metadata")

    stage = Usd.Stage.Open(str(lexical_layer), Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError("stage appearance layer could not be composed")
    composed_shader = stage.GetPrimAtPath(MATERIAL_PATH)
    if not composed_shader or composed_shader.GetTypeName() != "Shader":
        raise ValueError("composed stage appearance is missing source Shader prim")
    composed_diffuse = composed_shader.GetAttribute("inputs:diffuseColor")
    composed_opacity = composed_shader.GetAttribute("inputs:opacity")
    for sample in request["samples"]:
        time_code = float(sample["time_code"])
        expected_rgb = Gf.Vec3f(*[float(value) for value in sample["diffuse_color_linear"]])
        if composed_diffuse.Get(time_code) != expected_rgb:
            raise ValueError("composed stage appearance RGB does not match override")
        if composed_opacity.Get(time_code) != _float32(sample["opacity"]):
            raise ValueError("composed stage appearance opacity does not match override")

    stack_records = []
    for ordinal, stack_layer in enumerate(
        stage.GetLayerStack(includeSessionLayers=False), start=1
    ):
        if stack_layer.anonymous or not stack_layer.realPath:
            raise ValueError("stage appearance layer stack contains an anonymous layer")
        stack_path = _lexical_absolute(stack_layer.realPath)
        stack_snapshot = _regular_snapshot(stack_path, "composed USD layer")
        if ordinal == 1:
            if stack_path != lexical_layer:
                raise ValueError("stage appearance layer is not composed root layer")
        elif stack_path != twin and not stack_path.is_relative_to(twin):
            raise ValueError("stage appearance dependency is outside the public twin")
        stack_records.append(
            {
                "ordinal": ordinal,
                "resolved_path": str(stack_path),
                "sha256": stack_snapshot["sha256"],
                "bytes": stack_snapshot["bytes"],
            }
        )
    if len(stack_records) < 2 or Path(stack_records[1]["resolved_path"]) != source:
        raise ValueError("canonical source must be second in stage appearance layer stack")
    if _regular_snapshot(
        lexical_layer, "stage appearance layer", max_bytes=MAX_APPEARANCE_LAYER_BYTES
    ) != layer_snapshot:
        raise ValueError("stage appearance layer changed during inspection")
    if _regular_snapshot(source, "canonical fill-station source stage") != source_snapshot:
        raise ValueError("canonical source stage changed during inspection")
    return {
        "path": str(lexical_layer),
        "schema": STAGE_APPEARANCE_SCHEMA,
        "variant_id": variant_id,
        "stage_sha256": layer_snapshot["sha256"],
        "stage_bytes": layer_snapshot["bytes"],
        "source_stage": SOURCE_STAGE,
        "source_stage_sha256": source_snapshot["sha256"],
        "material_path": MATERIAL_PATH,
        "samples": request["samples"],
        "uncalibrated": True,
        "timing": layer_timing,
        "non_session_usd_layer_stack": stack_records,
    }


def create_stage_appearance_layer(
    twin_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    variant_id: object,
    samples: object,
) -> dict[str, object]:
    """Author, inspect, and no-replace publish one time-sampled override."""

    Gf, Sdf, Usd, Vt = _import_pxr()
    twin, source_before = _source_record(twin_root)
    source = Path(source_before["path"])
    _, timing = _validate_source_stage(source, source_before, Sdf, Usd)
    request = validate_stage_appearance_request(
        variant_id,
        samples,
        start_time_code=timing["start_time_code"],
        end_time_code=timing["end_time_code"],
    )
    output = _existing_directory(output_root, "output root")
    appearance_directory = output / "appearance-layers"
    if os.path.lexists(appearance_directory):
        _reject_symlink_components(appearance_directory, "appearance directory")
        if not appearance_directory.is_dir():
            raise ValueError("appearance directory must be a non-symlink directory")
    else:
        try:
            os.mkdir(appearance_directory, 0o755)
        except FileExistsError:
            _reject_symlink_components(appearance_directory, "appearance directory")
            if not appearance_directory.is_dir():
                raise ValueError("appearance directory must be a non-symlink directory")
        _fsync_directory(output)
    final_path = appearance_directory / f"{request['variant_id']}.usda"
    if os.path.lexists(final_path):
        raise FileExistsError(f"stage appearance output already exists: {final_path}")

    authored = Sdf.Layer.CreateAnonymous("stage-appearance.usda")
    authored.defaultPrim = "FillStation"
    authored.startTimeCode = timing["start_time_code"]
    authored.endTimeCode = timing["end_time_code"]
    authored.framesPerSecond = timing["frames_per_second"]
    authored.timeCodesPerSecond = timing["time_codes_per_second"]
    authored.subLayerPaths = [_canonical_relative_sublayer(final_path, source)]
    flattened_colors = [
        float(value)
        for sample in request["samples"]
        for value in sample["diffuse_color_linear"]
    ]
    authored.customLayerData = {
        "fluidmix_schema": STAGE_APPEARANCE_SCHEMA,
        "fluidmix_variant_id": request["variant_id"],
        "fluidmix_material_path": MATERIAL_PATH,
        "fluidmix_stage_time_codes": Vt.IntArray(
            [sample["time_code"] for sample in request["samples"]]
        ),
        "fluidmix_stage_diffuse_colors_linear": Vt.DoubleArray(flattened_colors),
        "fluidmix_stage_opacities": Vt.DoubleArray(
            [float(sample["opacity"]) for sample in request["samples"]]
        ),
        "fluidmix_source_stage": SOURCE_STAGE,
        "fluidmix_source_stage_sha256": source_before["sha256"],
        "fluidmix_uncalibrated": True,
    }
    for path in _EXPECTED_PRIM_PATHS:
        prim = Sdf.CreatePrimInLayer(authored, path)
        prim.specifier = Sdf.SpecifierOver
    shader = authored.GetPrimAtPath(MATERIAL_PATH)
    diffuse = Sdf.AttributeSpec(shader, "inputs:diffuseColor", Sdf.ValueTypeNames.Color3f)
    opacity_attribute = Sdf.AttributeSpec(shader, "inputs:opacity", Sdf.ValueTypeNames.Float)
    for sample in request["samples"]:
        time_code = float(sample["time_code"])
        authored.SetTimeSample(
            diffuse.path,
            time_code,
            Gf.Vec3f(*[float(value) for value in sample["diffuse_color_linear"]]),
        )
        authored.SetTimeSample(
            opacity_attribute.path, time_code, float(sample["opacity"])
        )
    content = authored.ExportToString().encode("utf-8")
    if len(content) > MAX_APPEARANCE_LAYER_BYTES:
        raise ValueError("authored stage appearance layer exceeds maximum size")

    temporary = appearance_directory / (
        f".{request['variant_id']}.{secrets.token_hex(16)}.tmp.usda"
    )
    published = False
    owned_identity: tuple[int, int] | None = None
    try:
        written = _write_exclusive(temporary, content)
        owned_identity = (written.st_dev, written.st_ino)
        inspect_stage_appearance_layer(
            temporary,
            twin,
            expected_variant_id=request["variant_id"],
            require_filename_match=False,
        )
        if _regular_snapshot(source, "canonical fill-station source stage") != source_before:
            raise ValueError("canonical source stage changed before publication")
        try:
            os.link(temporary, final_path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(
                f"stage appearance output already exists: {final_path}"
            ) from None
        published = True
        _require_owned_identity(final_path, owned_identity)
        _fsync_directory(appearance_directory)
        record = inspect_stage_appearance_layer(
            final_path, twin, expected_variant_id=request["variant_id"]
        )
        _require_owned_identity(final_path, owned_identity)
        if _regular_snapshot(source, "canonical fill-station source stage") != source_before:
            raise ValueError("canonical source stage changed during publication")
        _require_owned_identity(final_path, owned_identity)
        return record
    except Exception:
        if published and owned_identity is not None:
            try:
                final_metadata = os.lstat(final_path)
                if (final_metadata.st_dev, final_metadata.st_ino) == owned_identity:
                    os.unlink(final_path)
                    _fsync_directory(appearance_directory)
            except OSError:
                pass
        raise
    finally:
        try:
            temporary_metadata = os.lstat(temporary)
            if owned_identity is not None and (
                temporary_metadata.st_dev,
                temporary_metadata.st_ino,
            ) == owned_identity:
                os.unlink(temporary)
                _fsync_directory(appearance_directory)
        except FileNotFoundError:
            pass

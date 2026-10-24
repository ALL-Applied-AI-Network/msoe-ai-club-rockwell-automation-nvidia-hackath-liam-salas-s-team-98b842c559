"""Author and inspect neutral, uncalibrated USD appearance overrides.

Validation and import are deliberately independent of OpenUSD.  Functions that
open or author a USD layer import ``pxr`` only when called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import secrets
import stat
import struct


APPEARANCE_SCHEMA = "direct_appearance_v1"
MAX_APPEARANCE_LAYER_BYTES = 1_048_576
SOURCE_STAGE = "stations/fill_station/fill_station.usda"
MATERIAL_PATH = (
    "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"
)
_SAFE_VARIANT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMAND_PLAN_FIELDS = (
    "schema",
    "variant_id",
    "stage_sha256",
    "stage_bytes",
    "source_stage",
    "source_stage_sha256",
    "material_path",
    "diffuse_color_linear",
    "opacity",
)
_CUSTOM_LAYER_FIELDS = frozenset(
    {
        "fluidmix_schema",
        "fluidmix_variant_id",
        "fluidmix_material_path",
        "fluidmix_diffuse_color_linear",
        "fluidmix_opacity",
        "fluidmix_source_stage",
        "fluidmix_source_stage_sha256",
        "fluidmix_uncalibrated",
    }
)
_EXPECTED_PRIM_PATHS = (
    "/FillStation",
    "/FillStation/_materials",
    "/FillStation/_materials/USD_blue_liquid_approximation",
    MATERIAL_PATH,
)
_EXPECTED_LAYER_INFO_KEYS = frozenset(
    {
        "subLayerOffsets",
        "framesPerSecond",
        "customLayerData",
        "startTimeCode",
        "endTimeCode",
        "subLayers",
        "timeCodesPerSecond",
        "defaultPrim",
    }
)
_EXPECTED_ATTRIBUTE_INFO_KEYS = frozenset(
    {"typeName", "custom", "variability", "default"}
)


def _unit_interval_number(value: object, role: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{role} must be a non-Boolean number")
    if not math.isfinite(value):
        raise ValueError(f"{role} must be finite")
    if value < 0 or value > 1:
        raise ValueError(f"{role} must be in the closed interval [0, 1]")
    return value


def validate_appearance_request(
    variant_id: object, diffuse_color_linear: object, opacity: object
) -> dict[str, object]:
    """Validate neutral authoring values without clipping or coercing them."""

    if not isinstance(variant_id, str) or _SAFE_VARIANT_ID.fullmatch(variant_id) is None:
        raise ValueError(
            "variant ID must be 1-64 ASCII letters, digits, dot, underscore, or hyphen"
        )
    if (
        isinstance(diffuse_color_linear, (str, bytes))
        or not isinstance(diffuse_color_linear, Sequence)
        or len(diffuse_color_linear) != 3
    ):
        raise ValueError("linear RGB must contain exactly three numbers")
    rgb = [
        _unit_interval_number(value, f"linear RGB channel {index}")
        for index, value in enumerate(diffuse_color_linear)
    ]
    return {
        "variant_id": variant_id,
        "diffuse_color_linear": rgb,
        "opacity": _unit_interval_number(opacity, "opacity"),
    }


def appearance_command_plan(inspection: Mapping[str, object]) -> dict[str, object]:
    """Project an inspection into the exact preregistration object."""

    if not isinstance(inspection, Mapping):
        raise TypeError("appearance inspection must be a mapping")
    try:
        projected = {field: inspection[field] for field in _COMMAND_PLAN_FIELDS}
    except KeyError as error:
        raise ValueError(f"appearance inspection is missing {error.args[0]!r}") from error
    request = validate_appearance_request(
        projected["variant_id"],
        projected["diffuse_color_linear"],
        projected["opacity"],
    )
    if projected["schema"] != APPEARANCE_SCHEMA or type(projected["schema"]) is not str:
        raise ValueError("appearance inspection schema is invalid")
    for field in ("stage_sha256", "source_stage_sha256"):
        value = projected[field]
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"appearance inspection {field} is not a SHA-256")
    stage_bytes = projected["stage_bytes"]
    if (
        isinstance(stage_bytes, bool)
        or not isinstance(stage_bytes, int)
        or stage_bytes <= 0
        or stage_bytes > MAX_APPEARANCE_LAYER_BYTES
    ):
        raise ValueError("appearance inspection stage_bytes is outside the size bound")
    if projected["source_stage"] != SOURCE_STAGE or type(
        projected["source_stage"]
    ) is not str:
        raise ValueError("appearance inspection source stage is invalid")
    if projected["material_path"] != MATERIAL_PATH or type(
        projected["material_path"]
    ) is not str:
        raise ValueError("appearance inspection material path is invalid")
    projected["diffuse_color_linear"] = list(request["diffuse_color_linear"])
    projected["opacity"] = request["opacity"]
    return projected


def _import_pxr():
    try:
        from pxr import Gf, Sdf, Usd, Vt
    except ImportError as error:
        raise RuntimeError(
            "OpenUSD Python bindings are required to author or inspect appearance layers"
        ) from error
    return Gf, Sdf, Usd, Vt


def _lexical_absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _reject_symlink_components(path: Path, role: str) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"{role} path component is unavailable: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{role} must not contain a symlink path component: {current}")


def _existing_directory(path: str | os.PathLike[str], role: str) -> Path:
    candidate = _lexical_absolute(path)
    _reject_symlink_components(candidate, role)
    try:
        metadata = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"{role} must be an existing directory: {candidate}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{role} must be an existing non-symlink directory: {candidate}")
    return candidate


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _regular_snapshot(
    path: str | os.PathLike[str],
    role: str,
    *,
    max_bytes: int | None = None,
) -> dict[str, object]:
    candidate = _lexical_absolute(path)
    _reject_symlink_components(candidate, role)
    try:
        before = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"{role} is unavailable: {candidate}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file: {candidate}")
    if max_bytes is not None and before.st_size > max_bytes:
        raise ValueError(f"{role} exceeds the maximum size of {max_bytes} bytes")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ValueError(f"{role} could not be opened safely: {candidate}") from error
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode) or (
            opened_before.st_dev,
            opened_before.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise ValueError(f"{role} changed while it was opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if max_bytes is not None and byte_count > max_bytes:
                raise ValueError(f"{role} exceeds the maximum size of {max_bytes} bytes")
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(candidate)
    except OSError as error:
        raise ValueError(f"{role} changed while it was hashed") from error
    if (
        _identity(before) != _identity(opened_before)
        or _identity(opened_before) != _identity(opened_after)
        or _identity(opened_after) != _identity(after)
        or byte_count != after.st_size
    ):
        raise ValueError(f"{role} changed while it was hashed")
    return {
        "path": str(candidate),
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
    }


def _source_record(twin_root: str | os.PathLike[str]) -> tuple[Path, dict[str, object]]:
    twin = _existing_directory(twin_root, "public twin root")
    source = twin / SOURCE_STAGE
    return twin, _regular_snapshot(source, "canonical fill-station source stage")


def _validate_source_stage(source: Path, source_snapshot: Mapping[str, object], Sdf, Usd):
    layer = Sdf.Layer.FindOrOpen(str(source))
    if layer is None or not layer.Reload(force=True):
        raise ValueError("canonical fill-station source stage is not a readable USD layer")
    stage = Usd.Stage.Open(str(source), Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError("canonical fill-station source stage could not be composed")
    shader = stage.GetPrimAtPath(MATERIAL_PATH)
    if not shader or shader.GetTypeName() != "Shader":
        raise ValueError("canonical source is missing the required Shader prim")
    expected = {
        "inputs:diffuseColor": Sdf.ValueTypeNames.Color3f,
        "inputs:opacity": Sdf.ValueTypeNames.Float,
    }
    for name, value_type in expected.items():
        attribute = shader.GetAttribute(name)
        if not attribute or attribute.GetTypeName() != value_type or attribute.Get() is None:
            raise ValueError(
                f"canonical source shader property {name!r} is missing or has the wrong type"
            )
    if layer.defaultPrim != "FillStation":
        raise ValueError("canonical source default prim must be FillStation")
    timing = {
        "start_time_code": float(layer.startTimeCode),
        "end_time_code": float(layer.endTimeCode),
        "frames_per_second": float(layer.framesPerSecond),
        "time_codes_per_second": float(layer.timeCodesPerSecond),
    }
    if any(not math.isfinite(value) for value in timing.values()):
        raise ValueError("canonical source timing metadata must be finite")
    if _regular_snapshot(source, "canonical fill-station source stage") != source_snapshot:
        raise ValueError("canonical fill-station source stage changed during inspection")
    return layer, timing


def _all_prim_specs(layer) -> dict[str, object]:
    found: dict[str, object] = {}

    def visit(spec) -> None:
        found[str(spec.path)] = spec
        for child in spec.nameChildren:
            visit(child)

    for root in layer.rootPrims:
        visit(root)
    return found


def _float32(value: int | float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


def _canonical_relative_sublayer(layer_path: Path, source: Path) -> str:
    text = os.path.relpath(source, start=layer_path.parent)
    return PurePosixPath(*Path(text).parts).as_posix()


def inspect_appearance_layer(
    layer_path: str | os.PathLike[str],
    twin_root: str | os.PathLike[str],
    *,
    expected_variant_id: str | None = None,
    require_filename_match: bool = True,
) -> dict[str, object]:
    """Reopen and strictly inspect one direct-appearance root layer."""

    Gf, Sdf, Usd, Vt = _import_pxr()
    lexical_layer = _lexical_absolute(layer_path)
    if lexical_layer.suffix != ".usda":
        raise ValueError("appearance layer filename must end with .usda")
    layer_snapshot = _regular_snapshot(
        lexical_layer,
        "appearance layer",
        max_bytes=MAX_APPEARANCE_LAYER_BYTES,
    )
    twin, source_snapshot = _source_record(twin_root)
    source = Path(source_snapshot["path"])
    source_layer, source_timing = _validate_source_stage(
        source, source_snapshot, Sdf, Usd
    )

    layer = Sdf.Layer.FindOrOpen(str(lexical_layer))
    if layer is None or not layer.Reload(force=True):
        raise ValueError("appearance layer is not a readable USD layer")
    if set(layer.pseudoRoot.ListInfoKeys()) != _EXPECTED_LAYER_INFO_KEYS:
        raise ValueError("appearance layer contains missing or extra root metadata")
    custom = dict(layer.customLayerData)
    if set(custom) != _CUSTOM_LAYER_FIELDS:
        raise ValueError("appearance layer custom metadata fields are missing or extra")
    if custom["fluidmix_schema"] != APPEARANCE_SCHEMA or type(
        custom["fluidmix_schema"]
    ) is not str:
        raise ValueError("appearance layer schema metadata is invalid")
    variant_id = custom["fluidmix_variant_id"]
    rgb_raw = custom["fluidmix_diffuse_color_linear"]
    opacity = custom["fluidmix_opacity"]
    if not isinstance(rgb_raw, Vt.DoubleArray):
        raise ValueError("appearance layer RGB metadata must be a double array")
    request = validate_appearance_request(variant_id, list(rgb_raw), opacity)
    if expected_variant_id is not None and variant_id != expected_variant_id:
        raise ValueError("appearance layer variant metadata does not match the expected variant")
    if require_filename_match and lexical_layer.stem != variant_id:
        raise ValueError("appearance layer filename stem does not match variant metadata")
    if custom["fluidmix_material_path"] != MATERIAL_PATH or type(
        custom["fluidmix_material_path"]
    ) is not str:
        raise ValueError("appearance layer material-path metadata is invalid")
    if custom["fluidmix_source_stage"] != SOURCE_STAGE or type(
        custom["fluidmix_source_stage"]
    ) is not str:
        raise ValueError("appearance layer source-stage metadata is invalid")
    if custom["fluidmix_source_stage_sha256"] != source_snapshot["sha256"] or type(
        custom["fluidmix_source_stage_sha256"]
    ) is not str:
        raise ValueError("appearance layer source-stage hash does not match source bytes")
    if custom["fluidmix_uncalibrated"] is not True:
        raise ValueError("appearance layer must declare uncalibrated=true")

    if len(layer.subLayerPaths) != 1:
        raise ValueError("appearance layer must contain exactly one sublayer")
    if (
        len(layer.subLayerOffsets) != 1
        or layer.subLayerOffsets[0] != Sdf.LayerOffset()
    ):
        raise ValueError("appearance layer sublayer must use one identity layer offset")
    sublayer = layer.subLayerPaths[0]
    if (
        not isinstance(sublayer, str)
        or not sublayer
        or "\\" in sublayer
        or PurePosixPath(sublayer).is_absolute()
        or posixpath.normpath(sublayer) != sublayer
    ):
        raise ValueError("appearance layer sublayer must be one canonical relative path")
    expected_sublayer = _canonical_relative_sublayer(lexical_layer, source)
    if sublayer != expected_sublayer:
        raise ValueError("appearance layer sublayer does not resolve canonically to source")
    resolved_sublayer = (lexical_layer.parent / Path(*PurePosixPath(sublayer).parts)).resolve(
        strict=True
    )
    if resolved_sublayer != source:
        raise ValueError("appearance layer sublayer does not resolve to the exact source stage")
    if layer.defaultPrim != "FillStation":
        raise ValueError("appearance layer default prim must be FillStation")
    layer_timing = {
        "start_time_code": float(layer.startTimeCode),
        "end_time_code": float(layer.endTimeCode),
        "frames_per_second": float(layer.framesPerSecond),
        "time_codes_per_second": float(layer.timeCodesPerSecond),
    }
    if layer_timing != source_timing:
        raise ValueError("appearance layer timing does not exactly match the source stage")

    prims = _all_prim_specs(layer)
    if tuple(prims) != _EXPECTED_PRIM_PATHS:
        raise ValueError("appearance layer prim structure is missing, extra, or reordered")
    for path, prim in prims.items():
        if prim.specifier != Sdf.SpecifierOver or prim.typeName:
            raise ValueError("appearance layer may contain only untyped ancestor over specs")
        allowed_prim_info = {"specifier"}
        if set(prim.ListInfoKeys()) != allowed_prim_info:
            raise ValueError(f"appearance prim {path} contains unsupported authored metadata")
        expected_properties = (
            {"inputs:diffuseColor", "inputs:opacity"}
            if path == MATERIAL_PATH
            else set()
        )
        if {property_spec.name for property_spec in prim.properties} != expected_properties:
            raise ValueError("appearance layer has missing or extra shader properties")
    shader = prims[MATERIAL_PATH]
    diffuse = shader.attributes["inputs:diffuseColor"]
    opacity_attribute = shader.attributes["inputs:opacity"]
    if diffuse.typeName != Sdf.ValueTypeNames.Color3f:
        raise ValueError("appearance diffuseColor property must have type color3f")
    if opacity_attribute.typeName != Sdf.ValueTypeNames.Float:
        raise ValueError("appearance opacity property must have type float")
    expected_rgb32 = Gf.Vec3f(*(float(value) for value in request["diffuse_color_linear"]))
    if diffuse.default != expected_rgb32:
        raise ValueError("appearance diffuseColor property does not match exact metadata")
    if opacity_attribute.default != _float32(request["opacity"]):
        raise ValueError("appearance opacity property does not match exact metadata")
    for attribute in (diffuse, opacity_attribute):
        if (
            set(attribute.ListInfoKeys()) != _EXPECTED_ATTRIBUTE_INFO_KEYS
            or attribute.custom
            or attribute.variability != Sdf.VariabilityVarying
        ):
            raise ValueError("appearance shader property contains unsupported metadata")
        if attribute.connectionPathList.GetAddedOrExplicitItems():
            raise ValueError("appearance shader properties must not contain connections")
        if layer.ListTimeSamplesForPath(attribute.path):
            raise ValueError("appearance shader properties must not contain time samples")

    stage = Usd.Stage.Open(str(lexical_layer), Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError("appearance layer could not be composed")
    composed_shader = stage.GetPrimAtPath(MATERIAL_PATH)
    if not composed_shader or composed_shader.GetTypeName() != "Shader":
        raise ValueError("composed appearance stage is missing the source Shader prim")
    composed_diffuse = composed_shader.GetAttribute("inputs:diffuseColor")
    composed_opacity = composed_shader.GetAttribute("inputs:opacity")
    if (
        not composed_diffuse
        or composed_diffuse.GetTypeName() != Sdf.ValueTypeNames.Color3f
        or composed_diffuse.Get() != expected_rgb32
        or not composed_opacity
        or composed_opacity.GetTypeName() != Sdf.ValueTypeNames.Float
        or composed_opacity.Get() != _float32(request["opacity"])
    ):
        raise ValueError("composed appearance shader values do not match the override")
    layer_stack = stage.GetLayerStack(includeSessionLayers=False)
    if len(layer_stack) < 2:
        raise ValueError("appearance layer stack is incomplete")
    stack_records = []
    for ordinal, stack_layer in enumerate(layer_stack, start=1):
        if stack_layer.anonymous or not stack_layer.realPath:
            raise ValueError("appearance layer stack contains an anonymous layer")
        stack_path = _lexical_absolute(stack_layer.realPath)
        stack_snapshot = _regular_snapshot(stack_path, "composed USD layer")
        if ordinal == 1:
            if stack_path != lexical_layer:
                raise ValueError("appearance layer is not the composed root layer")
        else:
            if stack_path != twin and not stack_path.is_relative_to(twin):
                raise ValueError(
                    "appearance layer stack contains a dependency outside the public twin"
                )
        stack_records.append(
            {
                "ordinal": ordinal,
                "resolved_path": str(stack_path),
                "sha256": stack_snapshot["sha256"],
                "bytes": stack_snapshot["bytes"],
            }
        )
    if Path(stack_records[1]["resolved_path"]) != source:
        raise ValueError("canonical fill-station source must be second in the layer stack")
    if _regular_snapshot(
        lexical_layer, "appearance layer", max_bytes=MAX_APPEARANCE_LAYER_BYTES
    ) != layer_snapshot:
        raise ValueError("appearance layer changed during inspection")
    if _regular_snapshot(source, "canonical fill-station source stage") != source_snapshot:
        raise ValueError("canonical fill-station source stage changed during inspection")

    return {
        "path": str(lexical_layer),
        "schema": APPEARANCE_SCHEMA,
        "variant_id": variant_id,
        "stage_sha256": layer_snapshot["sha256"],
        "stage_bytes": layer_snapshot["bytes"],
        "source_stage": SOURCE_STAGE,
        "source_stage_sha256": source_snapshot["sha256"],
        "material_path": MATERIAL_PATH,
        "diffuse_color_linear": list(request["diffuse_color_linear"]),
        "opacity": request["opacity"],
        "uncalibrated": True,
        "timing": layer_timing,
        "non_session_usd_layer_stack": stack_records,
    }


def _write_exclusive(path: Path, content: bytes) -> os.stat_result:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    owned = os.fstat(descriptor)
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("appearance temp write made no progress")
            offset += written
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
        return os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            current = os.lstat(path)
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                os.unlink(path)
                _fsync_directory(path.parent)
        except FileNotFoundError:
            pass
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_owned_identity(path: Path, owned_identity: tuple[int, int]) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as error:
        raise ValueError("published appearance layer became unavailable") from error
    if not stat.S_ISREG(metadata.st_mode) or (
        metadata.st_dev,
        metadata.st_ino,
    ) != owned_identity:
        raise ValueError("published appearance layer identity was replaced")


def create_appearance_layer(
    twin_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    variant_id: object,
    diffuse_color_linear: object,
    opacity: object,
) -> dict[str, object]:
    """Author, inspect, and no-replace publish one appearance override."""

    request = validate_appearance_request(variant_id, diffuse_color_linear, opacity)
    Gf, Sdf, Usd, Vt = _import_pxr()
    twin, source_before = _source_record(twin_root)
    source = Path(source_before["path"])
    _, timing = _validate_source_stage(source, source_before, Sdf, Usd)
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
        raise FileExistsError(f"appearance output already exists: {final_path}")

    authored = Sdf.Layer.CreateAnonymous("appearance.usda")
    authored.defaultPrim = "FillStation"
    authored.startTimeCode = timing["start_time_code"]
    authored.endTimeCode = timing["end_time_code"]
    authored.framesPerSecond = timing["frames_per_second"]
    authored.timeCodesPerSecond = timing["time_codes_per_second"]
    authored.subLayerPaths = [_canonical_relative_sublayer(final_path, source)]
    authored.customLayerData = {
        "fluidmix_schema": APPEARANCE_SCHEMA,
        "fluidmix_variant_id": request["variant_id"],
        "fluidmix_material_path": MATERIAL_PATH,
        "fluidmix_diffuse_color_linear": Vt.DoubleArray(
            [float(value) for value in request["diffuse_color_linear"]]
        ),
        "fluidmix_opacity": float(request["opacity"]),
        "fluidmix_source_stage": SOURCE_STAGE,
        "fluidmix_source_stage_sha256": source_before["sha256"],
        "fluidmix_uncalibrated": True,
    }
    for path in _EXPECTED_PRIM_PATHS:
        prim = Sdf.CreatePrimInLayer(authored, path)
        prim.specifier = Sdf.SpecifierOver
    shader = authored.GetPrimAtPath(MATERIAL_PATH)
    diffuse = Sdf.AttributeSpec(
        shader, "inputs:diffuseColor", Sdf.ValueTypeNames.Color3f
    )
    diffuse.default = Gf.Vec3f(
        *(float(value) for value in request["diffuse_color_linear"])
    )
    opacity_attribute = Sdf.AttributeSpec(
        shader, "inputs:opacity", Sdf.ValueTypeNames.Float
    )
    opacity_attribute.default = float(request["opacity"])
    content = authored.ExportToString().encode("utf-8")
    if len(content) > MAX_APPEARANCE_LAYER_BYTES:
        raise ValueError("authored appearance layer exceeds the maximum size")

    temporary = appearance_directory / (
        f".{request['variant_id']}.{secrets.token_hex(16)}.tmp.usda"
    )
    published = False
    owned_identity: tuple[int, int] | None = None
    try:
        written = _write_exclusive(temporary, content)
        owned_identity = (written.st_dev, written.st_ino)
        inspect_appearance_layer(
            temporary,
            twin,
            expected_variant_id=request["variant_id"],
            require_filename_match=False,
        )
        if _regular_snapshot(source, "canonical fill-station source stage") != source_before:
            raise ValueError("canonical fill-station source stage changed before publication")
        try:
            os.link(temporary, final_path, follow_symlinks=False)
        except FileExistsError:
            raise FileExistsError(f"appearance output already exists: {final_path}") from None
        published = True
        _require_owned_identity(final_path, owned_identity)
        _fsync_directory(appearance_directory)
        record = inspect_appearance_layer(
            final_path, twin, expected_variant_id=request["variant_id"]
        )
        _require_owned_identity(final_path, owned_identity)
        if _regular_snapshot(source, "canonical fill-station source stage") != source_before:
            raise ValueError("canonical fill-station source stage changed during publication")
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

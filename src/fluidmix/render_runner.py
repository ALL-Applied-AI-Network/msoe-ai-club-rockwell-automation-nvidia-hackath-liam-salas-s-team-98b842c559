"""Argument-safe command builders for the bounded Isaac render batch."""

import hashlib
from pathlib import Path
import struct
from typing import Mapping
import zlib

from .render_manifest import RenderJob, RenderManifest


_ALLOWED_CHILD_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE",
    "TZ", "TMPDIR", "TEMP", "TMP", "XDG_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY",
    "LD_LIBRARY_PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_REQUIRE_CUDA", "NVIDIA_DISABLE_REQUIRE", "EXP_PATH",
    "ISAAC_PATH", "CARB_APP_PATH", "OMNI_KIT_ROOT", "OMNI_USER_FOLDER", "OV_CACHE_ROOT",
    "OV_DATA_ROOT", "OV_LOG_ROOT", "ACCEPT_EULA", "OMNI_KIT_ACCEPT_EULA",
)


def _number(value: float) -> str:
    return format(value, ".15g")


def build_variant_command(
    isaac_python: str,
    variant_script: str,
    twin: str,
    layer_path: str | Path,
    job: RenderJob,
) -> list[str] | None:
    """Build a direct-material layer command, or None for an unchanged-stage job."""
    if job.material is None:
        return None
    return [
        str(isaac_python),
        str(variant_script),
        "--twin",
        str(twin),
        "--material-color",
        ",".join(_number(value) for value in job.material.color),
        "--opacity",
        _number(job.material.opacity),
        "--variant-id",
        job.variant_id,
        "--out",
        str(layer_path),
    ]


def build_render_command(
    isaac_python: str,
    render_script: str,
    stage_path: str | Path,
    output_path: str | Path,
    manifest: RenderManifest,
    job: RenderJob,
) -> list[str]:
    """Build one fixed-FillCam render command without a shell."""
    command = [
        str(isaac_python),
        str(render_script),
        "--camera",
        manifest.camera,
        "--frame",
        str(job.frame),
        "--subframes",
        str(manifest.subframes),
        "--stage",
        str(stage_path),
        "--out",
        str(output_path),
    ]
    command.extend(f"--kit-arg={value}" for value in manifest.kit_args)
    return command


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_child_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Copy only runtime settings needed by Isaac; never forward arbitrary credentials."""
    result = {name: source[name] for name in _ALLOWED_CHILD_ENV if name in source}
    result["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    return result


def output_path_occupied(path: str | Path) -> bool:
    """Treat regular paths and dangling symlinks as occupied."""
    candidate = Path(path)
    return candidate.exists() or candidate.is_symlink()


def png_dimensions(path: str | Path) -> tuple[int, int]:
    """Validate a complete non-interlaced 8-bit RGB/RGBA PNG and return dimensions."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    offset = 8
    ihdr = None
    compressed = []
    seen_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError(f"truncated PNG chunk: {path}")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError(f"truncated PNG chunk data: {path}")
        chunk = data[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length:end])[0]
        actual_crc = zlib.crc32(kind + chunk) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"PNG chunk CRC mismatch: {path}")
        if kind == b"IHDR":
            if ihdr is not None or offset != 8 or length != 13:
                raise ValueError(f"invalid PNG IHDR: {path}")
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            compressed.append(chunk)
        elif kind == b"IEND":
            if length != 0 or end != len(data):
                raise ValueError(f"invalid PNG IEND: {path}")
            seen_iend = True
            break
        offset = end
    if ihdr is None or not compressed or not seen_iend:
        raise ValueError(f"incomplete PNG: {path}")
    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    if not width or not height or width > 16384 or height > 16384:
        raise ValueError(f"invalid PNG dimensions: {path}")
    if bit_depth != 8 or color_type not in (2, 6) or (compression, filtering, interlace) != (0, 0, 0):
        raise ValueError(f"unsupported PNG encoding: {path}")
    channels = 3 if color_type == 2 else 4
    expected_size = height * (1 + width * channels)
    if expected_size > 256 * 1024 * 1024:
        raise ValueError(f"PNG is too large to validate: {path}")
    try:
        pixels = zlib.decompress(b"".join(compressed))
    except zlib.error as exc:
        raise ValueError(f"invalid PNG image stream: {path}") from exc
    if len(pixels) != expected_size:
        raise ValueError(f"PNG image stream has the wrong size: {path}")
    row_size = 1 + width * channels
    if any(pixels[row * row_size] > 4 for row in range(height)):
        raise ValueError(f"PNG contains an invalid row filter: {path}")
    return width, height

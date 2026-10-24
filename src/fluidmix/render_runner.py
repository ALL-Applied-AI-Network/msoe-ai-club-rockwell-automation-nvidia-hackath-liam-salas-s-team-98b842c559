"""Safety checks and evidence inspection for one FillCam recovery render."""

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
from typing import Mapping
import zlib

from .render_manifest import RenderJob, RenderManifest


_ALLOWED_CHILD_ENV = (
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE",
    "TZ", "TMPDIR", "TEMP", "TMP", "XDG_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_REQUIRE_CUDA", "ACCEPT_EULA", "OMNI_KIT_ACCEPT_EULA",
)
ASYNC_THROTTLING_SETTING = "/exts/isaacsim.core.throttling/enable_async"
RECOVERY_KIT_ARGS = (
    "--/telemetry/enableAnonymousData=false",
    "--/structuredLog/enable=false",
    f"--{ASYNC_THROTTLING_SETTING}=false",
)
RECOVERY_SOURCE_PATHS = (
    "data/twin_render_recovery_v1.json",
    "scripts/render_fillcam_recovery.py",
    "scripts/run_twin_render_recovery.py",
    "src/fluidmix/render_manifest.py",
    "src/fluidmix/render_runner.py",
)


def require_async_throttling_disabled(value: object) -> None:
    """Validate the live Isaac async-throttling setting before capture."""
    if type(value) is not bool or value is not False:
        raise RuntimeError(
            "live /exts/isaacsim.core.throttling/enable_async must be Boolean false"
        )


def attest_live_async_setting(settings: object) -> dict[str, object]:
    """Read and validate the live Kit setting after extensions start."""
    value = settings.get(ASYNC_THROTTLING_SETTING)
    require_async_throttling_disabled(value)
    return {
        "path": ASYNC_THROTTLING_SETTING,
        "type": type(value).__name__,
        "value": value,
    }


def parse_live_setting_attestation(log_text: str) -> dict[str, object] | None:
    """Parse the renderer's single structured live-setting attestation line."""
    prefix = "live_setting_attestation "
    payloads = [line[len(prefix):] for line in log_text.splitlines() if line.startswith(prefix)]
    if not payloads:
        return None
    if len(payloads) != 1:
        raise ValueError("renderer log contains multiple live-setting attestations")
    try:
        record = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise ValueError("renderer live-setting attestation is not valid JSON") from exc
    expected_keys = {"path", "type", "value"}
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError("renderer live-setting attestation has the wrong fields")
    if record["path"] != ASYNC_THROTTLING_SETTING or record["type"] != "bool":
        raise ValueError("renderer live-setting attestation has the wrong path or type")
    require_async_throttling_disabled(record["value"])
    return record


def validate_recovery_render_request(
    twin: str | Path,
    stage: str | Path,
    camera: str,
    frame: int,
    subframes: int,
    resolution: tuple[int, int],
) -> Path:
    """Validate the one fixed-view recovery request and return its stage path."""
    if camera != "FillCam":
        raise ValueError("camera must be FillCam")
    if type(frame) is not int or frame != 686:
        raise ValueError("frame must be 686")
    if type(subframes) is not int or subframes != 32:
        raise ValueError("subframes must be 32")
    if resolution != (640, 480):
        raise ValueError("resolution must be 640x480")
    root = Path(twin).expanduser().resolve(strict=True)
    expected = (root / "stations/fill_station/fill_station.usda").resolve(strict=True)
    actual = Path(stage).expanduser().resolve(strict=True)
    if actual != expected:
        raise ValueError("stage must be the fixed fill-station file")
    return actual


def validate_recovery_kit_args(values: object) -> tuple[str, ...]:
    """Require the exact privacy flags plus the one reviewed renderer intervention."""
    if not isinstance(values, (list, tuple)) or tuple(values) != RECOVERY_KIT_ARGS:
        raise ValueError("Kit arguments must exactly match the reviewed recovery contract")
    if not all(type(value) is str for value in values):
        raise ValueError("Kit arguments must be strings")
    return RECOVERY_KIT_ARGS


def build_render_command(
    isaac_python: str,
    render_script: str,
    stage_path: str | Path,
    output_path: str | Path,
    manifest: RenderManifest,
    job: RenderJob,
    *,
    twin: str | Path,
) -> list[str]:
    """Build one fixed-FillCam render command without a shell."""
    validate_recovery_kit_args(manifest.kit_args)
    command = [
        str(isaac_python),
        str(render_script),
    ]
    command.extend(["--twin", str(twin)])
    command.extend([
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
    ])
    command.extend(f"--kit-arg={value}" for value in manifest.kit_args)
    return command


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"hash input must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError(f"could not open regular hash input without following: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"hash input must be a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda metadata: (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise ValueError(f"hash input changed while it was read: {path}")
    return digest.hexdigest()


def verify_source_lock(
    path: str | Path,
    *,
    expected_lock_sha256: str,
    root: str | Path,
) -> dict[str, object]:
    """Verify the externally pinned lock and every reviewed recovery source file."""
    if (
        type(expected_lock_sha256) is not str
        or len(expected_lock_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_lock_sha256)
    ):
        raise ValueError("expected source-lock digest must be lowercase SHA-256")
    lock_bytes, _ = _read_regular_snapshot(path, max_bytes=1024 * 1024)
    actual_lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    if actual_lock_sha256 != expected_lock_sha256:
        raise ValueError("source-lock digest does not match the externally pinned SHA-256")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"source lock contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(lock_bytes.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("source lock is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "files"}:
        raise ValueError("source lock must contain exactly schema_version and files")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("source lock schema_version must be exactly 1")
    files = payload["files"]
    if not isinstance(files, dict) or set(files) != set(RECOVERY_SOURCE_PATHS):
        raise ValueError("source lock files must be the exact recovery source set")

    root_candidate = Path(root)
    if root_candidate.is_symlink():
        raise ValueError("recovery source root must not be a symlink")
    base = root_candidate.resolve(strict=True)
    verified = {}
    for relative in RECOVERY_SOURCE_PATHS:
        expected = files[relative]
        if (
            type(expected) is not str
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(f"invalid source digest for {relative}")
        actual = sha256_file(base / relative)
        if actual != expected:
            raise ValueError(f"source digest mismatch for {relative}")
        verified[relative] = actual
    return {
        "lock_sha256": actual_lock_sha256,
        "files": verified,
    }


def hash_regular_tree(root: str | Path) -> dict[str, str]:
    """Hash every regular file under a symlink-free input tree."""
    candidate = Path(root)
    if candidate.is_symlink():
        raise ValueError(f"input tree root must not be a symlink: {root}")
    base = candidate.resolve(strict=True)
    if not base.is_dir():
        raise ValueError(f"input tree root must be a directory: {base}")
    result = {}
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"input tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"input tree contains a non-regular path: {path}")
        result[path.relative_to(base).as_posix()] = sha256_file(path)
    return result


def build_child_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Copy only runtime settings needed by Isaac; never forward arbitrary credentials."""
    result = {name: source[name] for name in _ALLOWED_CHILD_ENV if name in source}
    result["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    return result


def output_path_occupied(path: str | Path) -> bool:
    """Treat regular paths and dangling symlinks as occupied."""
    candidate = Path(path)
    return candidate.exists() or candidate.is_symlink()


def resolve_private_new_output(
    path: str | Path,
    *,
    expected_name: str | None = None,
) -> Path:
    """Bind a new output leaf to an existing private, symlink-free directory."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"output path must be absolute: {path}")
    if expected_name is not None and candidate.name != expected_name:
        raise ValueError(f"output name must be {expected_name!r}: {path}")
    if output_path_occupied(candidate):
        raise ValueError(f"refusing to overwrite output: {path}")
    lexical_parent = Path(os.path.abspath(candidate.parent))
    resolved_parent = candidate.parent.resolve(strict=True)
    if resolved_parent != lexical_parent:
        raise ValueError(f"output parent contains a symlink or traversal: {path}")
    if not resolved_parent.is_dir() or resolved_parent.stat().st_mode & 0o077:
        raise ValueError(f"output parent must be a private directory: {resolved_parent}")
    return resolved_parent / candidate.name


def _read_regular_snapshot(
    path: str | Path,
    *,
    max_bytes: int,
) -> tuple[bytes, os.stat_result]:
    """Read one regular file once without following a leaf symlink or accepting drift."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError(f"could not open input without following a symlink: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"input must be a regular file: {path}")
        if before.st_size > max_bytes:
            raise ValueError(f"input exceeds the {max_bytes}-byte limit: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"input exceeds the {max_bytes}-byte limit: {path}")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda metadata: (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if identity(before) != identity(after) or total != after.st_size:
        raise ValueError(f"input changed while it was read: {path}")
    return b"".join(chunks), after


def _png_rgb_statistics_data(
    data: bytes,
    *,
    expected_resolution: tuple[int, int],
    display_path: str | Path,
) -> dict[str, object]:
    """Decode one already-snapshotted RGB PNG and return explicit statistics."""
    path = display_path
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")

    offset = 8
    ihdr = None
    compressed = []
    seen_idat = False
    idat_ended = False
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
        if zlib.crc32(kind + chunk) & 0xFFFFFFFF != expected_crc:
            raise ValueError(f"PNG chunk CRC mismatch: {path}")
        if kind == b"IHDR":
            if ihdr is not None or offset != 8 or length != 13:
                raise ValueError(f"invalid PNG IHDR: {path}")
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            if ihdr is None or idat_ended:
                raise ValueError(f"PNG IDAT chunks are out of order: {path}")
            seen_idat = True
            compressed.append(chunk)
        elif kind == b"IEND":
            if not seen_idat or seen_iend or length != 0 or end != len(data):
                raise ValueError(f"invalid PNG IEND: {path}")
            seen_iend = True
            break
        else:
            if seen_idat:
                idat_ended = True
            if kind and kind[0] & 0x20 == 0:
                raise ValueError(f"unsupported critical PNG chunk {kind!r}: {path}")
        offset = end
    if ihdr is None or not compressed or not seen_iend:
        raise ValueError(f"incomplete PNG: {path}")

    width, height, bit_depth, color_type, compression, filtering, interlace = ihdr
    if (width, height) != expected_resolution:
        raise ValueError(
            f"PNG resolution {(width, height)} does not match {expected_resolution}: {path}"
        )
    if bit_depth != 8 or color_type != 2 or (compression, filtering, interlace) != (0, 0, 0):
        raise ValueError(f"PNG must be non-interlaced 8-bit RGB: {path}")
    row_bytes = width * 3
    expected_size = height * (row_bytes + 1)
    if expected_size > 256 * 1024 * 1024:
        raise ValueError(f"PNG decompressed data exceeds the 256 MiB limit: {path}")
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(b"".join(compressed), expected_size + 1)
        if decoder.unconsumed_tail or len(raw) > expected_size:
            raise ValueError(f"PNG image stream exceeds its declared size: {path}")
        raw += decoder.flush()
    except zlib.error as exc:
        raise ValueError(f"invalid PNG image stream: {path}") from exc
    if (
        len(raw) != expected_size
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError(f"PNG image stream has the wrong size or trailing data: {path}")

    channel_min = [255, 255, 255]
    channel_max = [0, 0, 0]
    channel_sum = [0, 0, 0]
    channel_sum_sq = [0, 0, 0]
    nonzero_pixels = 0
    distinct = set()
    position = 0
    prior = bytes(row_bytes)
    for _ in range(height):
        filter_type = raw[position]
        encoded = raw[position + 1:position + row_bytes + 1]
        position += row_bytes + 1
        if filter_type > 4:
            raise ValueError(f"invalid PNG row filter {filter_type}: {path}")
        row = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = row[index - 3] if index >= 3 else 0
            above = prior[index]
            upper_left = prior[index - 3] if index >= 3 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            else:
                estimate = left + above - upper_left
                left_distance = abs(estimate - left)
                above_distance = abs(estimate - above)
                upper_left_distance = abs(estimate - upper_left)
                predictor = (
                    left
                    if left_distance <= above_distance and left_distance <= upper_left_distance
                    else above
                    if above_distance <= upper_left_distance
                    else upper_left
                )
            row[index] = (value + predictor) & 0xFF
        for index in range(0, row_bytes, 3):
            pixel = (row[index], row[index + 1], row[index + 2])
            distinct.add(pixel)
            nonzero_pixels += int(any(pixel))
            for channel, value in enumerate(pixel):
                channel_min[channel] = min(channel_min[channel], value)
                channel_max[channel] = max(channel_max[channel], value)
                channel_sum[channel] += value
                channel_sum_sq[channel] += value * value
        prior = bytes(row)

    pixel_count = width * height
    means = [total / pixel_count for total in channel_sum]
    variance_numerators = [
        pixel_count * total_sq - total * total
        for total, total_sq in zip(channel_sum, channel_sum_sq)
    ]
    channel_stddev = [
        math.sqrt(numerator) / pixel_count for numerator in variance_numerators
    ]
    spatial_rgb_stddev = math.sqrt(
        sum(variance_numerators) / (3 * pixel_count * pixel_count)
    )
    return {
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "scalar_rgb_samples": pixel_count * 3,
        "channel_min": channel_min,
        "channel_max": channel_max,
        "channel_mean": means,
        "channel_stddev": channel_stddev,
        "spatial_rgb_stddev": spatial_rgb_stddev,
        "nonzero_rgb_pixels": nonzero_pixels,
        "nonzero_rgb_fraction": nonzero_pixels / pixel_count,
        "distinct_rgb_count": len(distinct),
    }


def png_rgb_statistics(
    path: str | Path,
    *,
    expected_resolution: tuple[int, int] = (640, 480),
) -> dict[str, object]:
    """Snapshot and decode one bounded RGB PNG with explicit pixel statistics."""
    data, _ = _read_regular_snapshot(path, max_bytes=16 * 1024 * 1024)
    return _png_rgb_statistics_data(
        data,
        expected_resolution=expected_resolution,
        display_path=path,
    )


def require_nonblank_rgb(statistics: Mapping[str, object]) -> None:
    """Reject known black or spatially constant RGB failure modes."""
    if (
        max(statistics["channel_max"]) == 0
        or statistics["nonzero_rgb_pixels"] == 0
    ):
        raise ValueError("RGB output is uniformly black")
    if (
        statistics["distinct_rgb_count"] < 2
        or statistics["spatial_rgb_stddev"] <= 0
        or not any(value > 0 for value in statistics["channel_stddev"])
    ):
        raise ValueError("RGB output is spatially constant")


def inspect_render_output(
    path: str | Path,
    *,
    expected_resolution: tuple[int, int] = (640, 480),
) -> dict[str, object]:
    """Inventory and mechanically inspect output without hiding partial evidence."""
    candidate = Path(path)
    is_symlink = candidate.is_symlink()
    exists = candidate.exists()
    is_regular = exists and candidate.is_file() and not is_symlink
    result = {
        "exists": exists,
        "is_symlink": is_symlink,
        "is_regular": is_regular,
    }
    if not is_regular:
        return result
    try:
        data, metadata = _read_regular_snapshot(
            candidate, max_bytes=16 * 1024 * 1024
        )
        statistics = _png_rgb_statistics_data(
            data,
            expected_resolution=expected_resolution,
            display_path=candidate,
        )
    except (OSError, ValueError) as exc:
        result.update(decoder_passed=False, decoder_error=str(exc))
        return result
    result.update(
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
        sha256=hashlib.sha256(data).hexdigest(),
        decoder_passed=True,
        statistics=statistics,
        evidence_snapshot="single_read",
    )
    try:
        require_nonblank_rgb(statistics)
    except ValueError as exc:
        result.update(content_gate_passed=False, content_error=str(exc))
    else:
        result["content_gate_passed"] = True
    return result

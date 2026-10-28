"""Render fail-closed FillCam RGB, semantic, instance, and liquid-mask evidence.

This keeps the verified Isaac Sim 6.0.1 FillCam recipe: PathTracing, no clipping-range
re-sync, warm-up updates, explicit time-code stepping, and bounded per-frame retries. Isaac
imports and ``SimulationApp`` startup occur only after CLI and filesystem preflight.

CPU tests cover argument handling, semantic planning, and captured array/mapping validity.
They cannot prove Isaac session-layer behavior, rendered geometry, or sim-to-real validity.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fluidmix.render_capture import (  # noqa: E402
    LIQUID_OVERRIDE,
    TRANSPARENT_ANNOTATION_OVERRIDE,
    CaptureValidation,
    TransparentAnnotationSourceFact,
    plan_semantic_assignments,
    plan_transparent_annotation_override,
    transparent_annotation_summary_record,
    validate_capture,
)


SUMMARY_SCHEMA_VERSION = 1
FAILURE_DIAGNOSTIC_ITEM_LIMIT = 64
FAILURE_DIAGNOSTIC_TEXT_LIMIT = 512
# Isaac Sim 6.0.1 Replicator troubleshooting documents frame skipping when the
# stopped timeline lets isaacsim.core.throttling toggle async rendering.
REPLICATOR_EXTRA_ARGS = (
    "--/exts/isaacsim.core.throttling/enable_async=false",
)
MIN_FILL_STATION_TIME_CODE = 1
MAX_FILL_STATION_TIME_CODE = 686
MIN_PATH_TRACING_BOUNCES = 1
MAX_PATH_TRACING_BOUNCES = 64
PATH_TRACING_BOUNCE_KEYS = {
    "max_bounces": "/rtx/pathtracing/maxBounces",
    "max_specular_transmission_bounces": (
        "/rtx/pathtracing/maxSpecularAndTransmissionBounces"
    ),
}


@dataclass(frozen=True)
class Preflight:
    twin: Path
    stage: Path
    camera_file: Path
    script: Path
    output: Path
    camera_record: dict[str, Any]
    resolution: tuple[int, int]
    frame_range: tuple[int, ...]
    time_codes: tuple[int, ...]
    input_snapshots: tuple[FileSnapshot, ...]


@dataclass(frozen=True)
class FileSnapshot:
    role: str
    path: Path
    sha256: str
    size_bytes: int


class InputDriftError(RuntimeError):
    """An input file no longer matches its pre-capture byte snapshot."""


def snapshot_input_file(role: str, path: Path) -> FileSnapshot:
    """Hash one resolved input before capture."""

    if not isinstance(role, str) or not role:
        raise ValueError("input snapshot role must be a nonempty string")
    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        if not resolved.is_file():
            raise InputDriftError(f"input {role} is not a regular file: {resolved}")
        digest = sha256_file(resolved)
        after = resolved.stat()
    except InputDriftError:
        raise
    except OSError as error:
        raise InputDriftError(f"input {role} is unavailable: {path}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise InputDriftError(f"input {role} changed while hashing: {resolved}")
    return FileSnapshot(
        role=role,
        path=resolved,
        sha256=digest,
        size_bytes=after.st_size,
    )


def verify_input_snapshots(snapshots: tuple[FileSnapshot, ...]) -> None:
    """Reject missing or byte-drifted inputs before publication."""

    for expected in snapshots:
        actual = snapshot_input_file(expected.role, expected.path)
        if (
            actual.path != expected.path
            or actual.size_bytes != expected.size_bytes
            or actual.sha256 != expected.sha256
        ):
            raise InputDriftError(
                f"input {expected.role} drifted after snapshot: {expected.path}"
            )


@dataclass(frozen=True)
class SharedCaptureAttempt:
    attempts: int
    rgb: object
    semantic: object
    instance: object
    validation: CaptureValidation


@dataclass(frozen=True)
class CaptureRun:
    summary: dict[str, object]
    input_snapshots: tuple[FileSnapshot, ...]


class AnnotatorEnvelopeError(RuntimeError):
    """A retryable malformed semantic or instance annotator result."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


def _bounded_int_values(values: tuple[int, ...]) -> tuple[list[int], int, bool]:
    items = list(values[:FAILURE_DIAGNOSTIC_ITEM_LIMIT])
    return items, len(values), len(values) > len(items)


def _bounded_diagnostic_text(value: str) -> tuple[str, bool]:
    return value[:FAILURE_DIAGNOSTIC_TEXT_LIMIT], len(value) > FAILURE_DIAGNOSTIC_TEXT_LIMIT


def _validation_failure_summary(validation: CaptureValidation) -> dict[str, object]:
    """Return bounded validation facts without retaining rejected arrays."""

    def shape_dtype(
        stats: object, *, include_scalar_stats: bool = False
    ) -> dict[str, object] | None:
        if stats is None:
            return None
        summary = {"shape": list(stats.shape), "dtype": stats.dtype}
        if include_scalar_stats:
            summary.update(
                {
                    "minimum": stats.minimum,
                    "maximum": stats.maximum,
                    "mean": stats.mean,
                    "std": stats.std,
                }
            )
        return summary

    semantic_ids, semantic_id_count, semantic_ids_truncated = _bounded_int_values(
        validation.semantic_ids
    )
    instance_ids, instance_id_count, instance_ids_truncated = _bounded_int_values(
        validation.instance_ids
    )
    selected_ids, selected_id_count, selected_ids_truncated = _bounded_int_values(
        validation.selected_liquid_instance_ids
    )
    semantic_mapping = []
    if validation.class_mapping is not None:
        for record in validation.class_mapping.records[:FAILURE_DIAGNOSTIC_ITEM_LIMIT]:
            labels = [
                _bounded_diagnostic_text(label)[0]
                for label in record.labels[:FAILURE_DIAGNOSTIC_ITEM_LIMIT]
            ]
            semantic_mapping.append(
                {
                    "id": record.semantic_id,
                    "labels": labels,
                    "label_count": len(record.labels),
                    "labels_truncated": len(record.labels) > len(labels)
                    or any(
                        _bounded_diagnostic_text(label)[1]
                        for label in record.labels[:FAILURE_DIAGNOSTIC_ITEM_LIMIT]
                    ),
                }
            )
        semantic_mapping_count = len(validation.class_mapping.records)
    else:
        semantic_mapping_count = 0
    instance_mapping = []
    for record in validation.instance_mapping[:FAILURE_DIAGNOSTIC_ITEM_LIMIT]:
        bounded_path, path_truncated = _bounded_diagnostic_text(record.path)
        instance_mapping.append(
            {
                "id": record.instance_id,
                "path": bounded_path,
                "path_truncated": path_truncated,
            }
        )
    class_pixel_counts = [
        {
            "class": record.class_name,
            "ids": list(record.class_ids[:FAILURE_DIAGNOSTIC_ITEM_LIMIT]),
            "id_count": len(record.class_ids),
            "ids_truncated": len(record.class_ids) > FAILURE_DIAGNOSTIC_ITEM_LIMIT,
            "pixels": record.pixel_count,
        }
        for record in validation.class_pixel_counts[:FAILURE_DIAGNOSTIC_ITEM_LIMIT]
    ]
    return {
        "raster_shape": list(validation.raster_shape),
        "rgb": shape_dtype(validation.rgb, include_scalar_stats=True),
        "semantic": shape_dtype(validation.semantic),
        "instance": shape_dtype(validation.instance),
        "semantic_ids": semantic_ids,
        "semantic_id_count": semantic_id_count,
        "semantic_ids_truncated": semantic_ids_truncated,
        "instance_ids": instance_ids,
        "instance_id_count": instance_id_count,
        "instance_ids_truncated": instance_ids_truncated,
        "semantic_mapping_records": semantic_mapping,
        "semantic_mapping_record_count": semantic_mapping_count,
        "semantic_mapping_truncated": semantic_mapping_count > len(semantic_mapping),
        "instance_mapping_records": instance_mapping,
        "instance_mapping_record_count": len(validation.instance_mapping),
        "instance_mapping_truncated": len(validation.instance_mapping)
        > len(instance_mapping),
        "class_pixel_counts": class_pixel_counts,
        "selected_liquid_instance_ids": selected_ids,
        "selected_liquid_instance_id_count": selected_id_count,
        "selected_liquid_instance_ids_truncated": selected_ids_truncated,
        "liquid_instance_pixel_count": validation.liquid_instance_pixel_count,
        "instance_agreement": validation.instance_agreement,
    }


class CaptureRetryExhausted(RuntimeError):
    """All bounded attempts failed the shared-capture validation contract."""

    def __init__(
        self,
        *,
        attempt_denominator: int,
        last_reason_codes: tuple[str, ...],
        last_validation: CaptureValidation | None,
    ):
        self.attempt_denominator = attempt_denominator
        self.last_reason_codes = last_reason_codes
        self.last_validation_summary = (
            _validation_failure_summary(last_validation)
            if last_validation is not None
            else None
        )
        super().__init__(
            f"shared capture failed after {attempt_denominator} attempts: "
            f"{last_reason_codes!r}"
        )


@dataclass
class CaptureFailureContext:
    phase: str = "runtime_imports"
    requested_time_code: int | None = None


class FailureDiagnosticWriteError(RuntimeError):
    """A capture failed and its durable private diagnostic could not be written."""


def _failure_record(
    error: BaseException, context: CaptureFailureContext
) -> dict[str, object]:
    if isinstance(error, CaptureRetryExhausted):
        safe_message = str(error)
        attempt_denominator: int | None = error.attempt_denominator
        last_reason_codes = list(error.last_reason_codes)
        last_validation = error.last_validation_summary
    else:
        safe_message = f"{type(error).__name__} during {context.phase}"
        attempt_denominator = None
        last_reason_codes = []
        last_validation = None
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "mechanical_gates_passed": False,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "phase": context.phase,
        "requested_time_code": context.requested_time_code,
        "exception": {
            "type": type(error).__name__,
            "message": safe_message,
        },
        "attempt_denominator": attempt_denominator,
        "last_reason_codes": last_reason_codes,
        "last_validation": last_validation,
    }


def _write_json_atomic_durable(path: Path, record: dict[str, object]) -> None:
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=os.fspath(path.parent)
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor_open = False
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if descriptor_open:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _emit_failure_record(record: dict[str, object], stderr: object) -> None:
    stderr.write(
        json.dumps(
            {"event": "twin_render_failure", "failure": record},
            sort_keys=True,
        )
        + "\n"
    )
    stderr.flush()


@contextmanager
def _diagnostic_kit_lifecycle(
    *,
    kit: object,
    partial: Path,
    stderr: object | None = None,
):
    """Close Kit while preserving structured failure evidence first."""

    context = CaptureFailureContext()
    destination = sys.stderr if stderr is None else stderr
    try:
        yield context
    except BaseException as error:
        record = _failure_record(error, context)
        diagnostic_error: FailureDiagnosticWriteError | None = None
        try:
            _write_json_atomic_durable(partial / "failure.json", record)
        except BaseException as write_error:
            diagnostic_error = FailureDiagnosticWriteError(
                "failure.json could not be written durably: "
                f"{type(write_error).__name__}"
            )
            record["diagnostic_write"] = {
                "status": "failed",
                "exception_type": type(write_error).__name__,
                "message": "failure.json could not be written durably",
            }
        try:
            _emit_failure_record(record, destination)
        except BaseException as emit_error:
            if diagnostic_error is None:
                diagnostic_error = FailureDiagnosticWriteError(
                    "structured failure stderr could not be emitted and flushed: "
                    f"{type(emit_error).__name__}"
                )
        kit.close(exit_code=1)
        if diagnostic_error is not None:
            raise diagnostic_error from error
        raise
    else:
        kit.close()


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def bounded_pathtracing_bounces(value: str) -> int:
    """Parse an explicit, bounded RTX path-tracing bounce override."""

    parsed = positive_int(value)
    if parsed > MAX_PATH_TRACING_BOUNCES:
        raise argparse.ArgumentTypeError(
            f"must be between {MIN_PATH_TRACING_BOUNCES} and {MAX_PATH_TRACING_BOUNCES}"
        )
    return parsed


def pathtracing_bounce_settings(args: argparse.Namespace) -> dict[str, int]:
    """Return only explicitly requested RTX settings; defaults remain untouched."""

    settings: dict[str, int] = {}
    for argument, key in PATH_TRACING_BOUNCE_KEYS.items():
        value = getattr(args, argument)
        if value is not None:
            settings[key] = value
    return settings


def pathtracing_bounce_summary(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    """Return the exact optional bounce-override record stored in a summary."""

    return {"pathtracing_bounce_overrides": pathtracing_bounce_settings(args)}


def parse_frame_spec(value: str) -> tuple[int, ...]:
    """Parse a range or an explicit increasing Fill Station time-code list."""

    if not isinstance(value, str):
        raise ValueError("frame specification must be a string")
    if not value or (":" in value and "," in value):
        raise ValueError("frames must be a range or explicit list")
    if ":" in value:
        pieces = value.split(":")
        if len(pieces) != 3:
            raise ValueError("frames range must use start:stop:step syntax")
        try:
            start, stop, step = (int(piece) for piece in pieces)
        except ValueError as error:
            raise ValueError("frame start, stop, and step must be integers") from error
        if stop <= start:
            raise ValueError("frame stop must be greater than start")
        if step <= 0:
            raise ValueError("frame step must be positive")
        time_codes = tuple(range(start, stop, step))
    else:
        pieces = value.split(",")
        if any(not piece for piece in pieces):
            raise ValueError("explicit frame list contains an empty item")
        try:
            time_codes = tuple(int(piece) for piece in pieces)
        except ValueError as error:
            raise ValueError("explicit frame list must contain integers") from error
        if any(left >= right for left, right in zip(time_codes, time_codes[1:])):
            raise ValueError("explicit frame list must be unique and strictly increasing")
    if not time_codes:
        raise ValueError("frame specification selects no time codes")
    if any(
        code < MIN_FILL_STATION_TIME_CODE or code > MAX_FILL_STATION_TIME_CODE
        for code in time_codes
    ):
        raise ValueError("frame time code is outside the Fill Station stage range")
    return time_codes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twin", required=True, help="csi_digital_twin_v1 folder")
    parser.add_argument(
        "--stage",
        default="stations/fill_station/fill_station.usda",
        help="stage path; relative paths resolve under --twin",
    )
    parser.add_argument(
        "--camera",
        choices=("FillCam",),
        default="FillCam",
        help="production capture is restricted to the reviewed FillCam view",
    )
    parser.add_argument(
        "--frames",
        default="1:687:15",
        help="exclusive start:stop:step range or increasing comma-separated time codes",
    )
    parser.add_argument("--subframes", type=positive_int, default=32)
    parser.add_argument(
        "--renderer",
        default="PathTracing",
        choices=("PathTracing", "RaytracedLighting"),
    )
    parser.add_argument(
        "--max-bounces",
        type=bounded_pathtracing_bounces,
        default=None,
        help=(
            "optional /rtx/pathtracing/maxBounces override; omitted leaves the "
            "renderer default unchanged"
        ),
    )
    parser.add_argument(
        "--max-specular-transmission-bounces",
        type=bounded_pathtracing_bounces,
        default=None,
        help=(
            "optional /rtx/pathtracing/maxSpecularAndTransmissionBounces override; "
            "omitted leaves the renderer default unchanged"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=60,
        help="kit.update() calls before the first capture",
    )
    parser.add_argument(
        "--max-tries",
        type=positive_int,
        default=20,
        help="shared-capture retries per frame",
    )
    parser.add_argument(
        "--seed",
        type=nonnegative_int,
        default=20261024,
        help="Replicator global seed; full renderer determinism is not claimed",
    )
    parser.add_argument(
        "--require-liquid-pixels",
        action="store_true",
        help="require both liquid and vial semantic pixels for every kept frame",
    )
    parser.add_argument("--out", required=True)
    return parser


def validate_output_destination(path: os.PathLike[str] | str) -> None:
    """Reject every occupied exact output path, including dangling symlinks."""

    candidate = os.fspath(path)
    if os.path.lexists(candidate):
        raise FileExistsError(f"output path already exists: {candidate}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_resolving_leaf(path: str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(path)))


def preflight(args: argparse.Namespace) -> Preflight:
    time_codes = parse_frame_spec(args.frames)
    frame_range = time_codes

    output = _absolute_without_resolving_leaf(args.out)
    validate_output_destination(output)

    twin = Path(args.twin).expanduser().resolve(strict=True)
    if not twin.is_dir():
        raise NotADirectoryError(f"twin root is not a directory: {twin}")
    requested_stage = Path(args.stage).expanduser()
    stage = (requested_stage if requested_stage.is_absolute() else twin / requested_stage).resolve(
        strict=True
    )
    if not stage.is_file():
        raise FileNotFoundError(f"stage is not a file: {stage}")
    camera_file = (twin / "cameras" / f"{args.camera}.json").resolve(strict=True)
    if not camera_file.is_file():
        raise FileNotFoundError(f"camera record is not a file: {camera_file}")
    script = Path(__file__).resolve(strict=True)
    input_snapshots = (
        snapshot_input_file("stage", stage),
        snapshot_input_file("camera", camera_file),
        snapshot_input_file("script", script),
    )
    with camera_file.open("r", encoding="utf-8") as stream:
        camera_record = json.load(stream)
    verify_input_snapshots((input_snapshots[1],))
    render_px = camera_record.get("render_px")
    if (
        not isinstance(render_px, list)
        or len(render_px) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in render_px
        )
    ):
        raise ValueError(f"camera render_px must be two positive integers: {render_px!r}")
    for key in ("prim_in_csi_line", "prim_in_station_file"):
        value = camera_record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"camera record has invalid {key}: {value!r}")

    return Preflight(
        twin=twin,
        stage=stage,
        camera_file=camera_file,
        script=script,
        output=output,
        camera_record=camera_record,
        resolution=(render_px[0], render_px[1]),
        frame_range=frame_range,
        time_codes=time_codes,
        input_snapshots=input_snapshots,
    )


def _create_partial_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_output_destination(output)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=os.fspath(output.parent))
    )
    for name in ("rgb", "semantic", "instance", "liquid"):
        (partial / name).mkdir()
    return partial


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination is absent."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            "atomic renameat2(RENAME_NOREPLACE) is unavailable; refusing publication",
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number, os.strerror(error_number), os.fspath(destination)
        )
    raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def publish_partial_no_clobber(
    partial: Path,
    output: Path,
    *,
    rename_noreplace: Callable[[Path, Path], None] | None = None,
) -> None:
    """Publish a completed partial directory without replacing a competitor."""

    publisher = rename_noreplace or _atomic_rename_noreplace
    publisher(partial, output)


def _annotator_data_and_mapping(raw: object, *, name: str) -> tuple[object, object]:
    reason_code = f"{name}_annotator_envelope_invalid"
    if not isinstance(raw, dict):
        raise AnnotatorEnvelopeError(
            reason_code, "annotator did not return a data/info mapping"
        )
    if "data" not in raw:
        raise AnnotatorEnvelopeError(reason_code, "annotator result has no data")
    info = raw.get("info")
    if not isinstance(info, dict) or "idToLabels" not in info:
        raise AnnotatorEnvelopeError(
            reason_code, "annotator result has no info.idToLabels"
        )
    return raw["data"], info["idToLabels"]


def _retry_shared_capture(
    *,
    step: Callable[[], None],
    get_rgb: Callable[[], object],
    get_semantic: Callable[[], object],
    get_instance: Callable[[], object],
    np_module: object,
    max_tries: int,
    raster_shape: tuple[int, int],
    require_liquid_pixels: bool,
) -> SharedCaptureAttempt:
    """Retry one shared RGB/semantic/instance capture until all gates pass."""

    last_reason_codes: tuple[str, ...] = ()
    last_validation: CaptureValidation | None = None
    for attempt in range(1, max_tries + 1):
        step()
        raw_rgb = get_rgb()
        if (
            isinstance(raw_rgb, np_module.ndarray)
            and raw_rgb.ndim == 3
            and raw_rgb.shape[2] == 4
        ):
            capture_rgb = raw_rgb[..., :3]
        else:
            capture_rgb = raw_rgb
        try:
            semantic, semantic_mapping = _annotator_data_and_mapping(
                get_semantic(), name="semantic"
            )
            instance, instance_mapping = _annotator_data_and_mapping(
                get_instance(), name="instance"
            )
        except AnnotatorEnvelopeError as error:
            last_reason_codes = (error.reason_code,)
            continue
        validation = validate_capture(
            capture_rgb,
            semantic,
            instance,
            semantic_mapping,
            instance_mapping,
            raster_shape=raster_shape,
            require_liquid_pixels=require_liquid_pixels,
        )
        last_validation = validation
        last_reason_codes = validation.reason_codes
        if validation.valid:
            return SharedCaptureAttempt(
                attempts=attempt,
                rgb=capture_rgb,
                semantic=semantic,
                instance=instance,
                validation=validation,
            )
    raise CaptureRetryExhausted(
        attempt_denominator=max_tries,
        last_reason_codes=last_reason_codes,
        last_validation=last_validation,
    )


def _stage_layer_records(
    stage: object,
) -> tuple[list[dict[str, object]], tuple[FileSnapshot, ...]]:
    records: list[dict[str, object]] = []
    snapshots: list[FileSnapshot] = []
    for ordinal, layer in enumerate(stage.GetLayerStack(includeSessionLayers=False), start=1):
        real_path = str(layer.realPath) if layer.realPath else None
        record: dict[str, object] = {
            "ordinal": ordinal,
            "identifier": str(layer.identifier),
            "resolved_path": None,
            "sha256": None,
            "bytes": None,
        }
        if real_path:
            resolved = Path(real_path).resolve(strict=True)
            if not resolved.is_file():
                raise RuntimeError(f"non-session USD layer is not a file: {resolved}")
            snapshot = snapshot_input_file(f"usd_layer_{ordinal}", resolved)
            snapshots.append(snapshot)
            record.update(
                {
                    "resolved_path": str(snapshot.path),
                    "sha256": snapshot.sha256,
                    "bytes": snapshot.size_bytes,
                }
            )
        records.append(record)
    if not records:
        raise RuntimeError("opened stage reported no non-session layer stack")
    return records, tuple(snapshots)


def _collect_semantic_plan(stage: object, Usd: object, UsdSemantics: object):
    source_classes: dict[str, object] = {}
    prim_types: dict[str, str] = {}
    existing_effective: dict[str, tuple[str, ...]] = {}
    query = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default())
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        prim_types[path] = prim.GetTypeName()
        source = prim.GetAttribute("userProperties:synthetic_class")
        source_classes[path] = source.Get() if source and source.HasValue() else None
        existing_effective[path] = tuple(
            str(label) for label in query.ComputeUniqueInheritedLabels(prim)
        )
    return plan_semantic_assignments(source_classes, prim_types, existing_effective)


def _collect_transparent_annotation_plan(stage: object, semantic_plan: object):
    override = TRANSPARENT_ANNOTATION_OVERRIDE
    parent = stage.GetPrimAtPath(override.parent_path)
    mesh = stage.GetPrimAtPath(override.prim_path)
    prim_types: dict[str, str] = {}
    if parent.IsValid():
        prim_types[override.parent_path] = parent.GetTypeName()
    if mesh.IsValid():
        prim_types[override.prim_path] = mesh.GetTypeName()
    planned_classes = {
        assignment.path: assignment.class_name for assignment in semantic_plan.assignments
    }
    attribute = mesh.GetAttribute(override.attribute_name) if mesh.IsValid() else None
    attribute_exists = bool(attribute and attribute.IsValid())
    has_authored_value = bool(
        attribute_exists and attribute.HasAuthoredValueOpinion()
    )
    source = TransparentAnnotationSourceFact(
        attribute_exists=attribute_exists,
        type_name=str(attribute.GetTypeName()) if attribute_exists else None,
        has_authored_value=has_authored_value,
        value=attribute.Get() if attribute_exists else None,
    )
    return plan_transparent_annotation_override(
        prim_types,
        planned_classes,
        source,
    )


def _author_semantic_plan(stage: object, plan: object, Usd: object, UsdSemantics: object) -> None:
    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, session_layer):
        for assignment in plan.assignments:
            prim = stage.GetPrimAtPath(assignment.path)
            if not prim.IsValid():
                raise RuntimeError(f"semantic assignment prim disappeared: {assignment.path}")
            api = UsdSemantics.LabelsAPI.Apply(prim, "class")
            if not api:
                raise RuntimeError(f"could not apply class LabelsAPI at {assignment.path}")
            attribute = api.CreateLabelsAttr()
            if not attribute.Set([assignment.class_name]):
                raise RuntimeError(f"could not author class label at {assignment.path}")

        query = UsdSemantics.LabelsQuery("class", Usd.TimeCode.Default())
        for path in (LIQUID_OVERRIDE.root_path, LIQUID_OVERRIDE.mesh_path):
            effective = {
                str(label)
                for label in query.ComputeUniqueInheritedLabels(stage.GetPrimAtPath(path))
            }
            if effective != {LIQUID_OVERRIDE.class_name}:
                raise RuntimeError(
                    f"liquid override verification failed at {path}: {sorted(effective)!r}"
                )
    if not session_layer.dirty:
        raise RuntimeError("semantic override did not dirty the session layer")


def _author_transparent_annotation_override(
    stage: object,
    plan: object,
    Usd: object,
    Sdf: object,
) -> dict[str, object]:
    override = TRANSPARENT_ANNOTATION_OVERRIDE
    if plan.override != override:
        raise RuntimeError("transparent annotation plan does not use the fixed override")
    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, session_layer):
        mesh = stage.GetPrimAtPath(override.prim_path)
        if not mesh.IsValid() or mesh.GetTypeName() != "Mesh":
            raise RuntimeError(
                f"transparent annotation mesh disappeared: {override.prim_path}"
            )
        attribute = mesh.CreateAttribute(
            override.attribute_name,
            Sdf.ValueTypeNames.Bool,
            custom=True,
        )
        if not attribute.Set(override.authored_value):
            raise RuntimeError(
                f"could not author {override.attribute_name} at {override.prim_path}"
            )

    composed = stage.GetPrimAtPath(override.prim_path).GetAttribute(
        override.attribute_name
    )
    property_stack = composed.GetPropertyStack(Usd.TimeCode.Default())
    if not property_stack or property_stack[0].layer != session_layer:
        raise RuntimeError(
            f"transparent annotation override is not strongest in the session layer: "
            f"{override.prim_path}.{override.attribute_name}"
        )
    return transparent_annotation_summary_record(
        plan,
        composed_type_name=str(composed.GetTypeName()),
        composed_value=composed.Get(),
    )


def _write_png_checked(cv2: object, path: Path, array: object) -> None:
    if not cv2.imwrite(str(path), array):
        raise OSError(f"image writer failed: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.shape != array.shape or decoded.dtype != array.dtype:
        raise OSError(f"image write verification failed: {path}")
    if not (decoded == array).all():
        raise OSError(f"image write was not lossless: {path}")


def _write_npy_checked(np: object, path: Path, array: object) -> None:
    np.save(path, array, allow_pickle=False)
    decoded = np.load(path, allow_pickle=False)
    if decoded.shape != array.shape or decoded.dtype != array.dtype:
        raise OSError(f"array write verification failed: {path}")
    if not np.array_equal(decoded, array):
        raise OSError(f"array write was not lossless: {path}")


def _stats_record(stats: object) -> dict[str, object]:
    if stats is None:
        raise RuntimeError("valid capture is missing array statistics")
    record = asdict(stats)
    record["shape"] = list(record["shape"])
    return record


def _frame_record(
    *,
    validation: CaptureValidation,
    ordinal: int,
    time_code: int,
    attempts: int,
    seconds: float,
    artifacts: dict[str, dict[str, str]],
) -> dict[str, object]:
    if validation.class_mapping is None:
        raise RuntimeError("valid capture is missing a semantic mapping")
    return {
        "capture_ordinal": ordinal,
        "time_code": time_code,
        "attempts": attempts,
        "capture_seconds": round(seconds, 6),
        "reason_codes": list(validation.reason_codes),
        "rgb": _stats_record(validation.rgb),
        "semantic": _stats_record(validation.semantic),
        "instance": _stats_record(validation.instance),
        "semantic_ids": list(validation.semantic_ids),
        "instance_ids": list(validation.instance_ids),
        "class_ids": {
            "liquid": list(validation.class_mapping.liquid_ids),
            "vial": list(validation.class_mapping.vial_ids),
        },
        "class_pixel_counts": {
            record.class_name: record.pixel_count
            for record in validation.class_pixel_counts
        },
        "semantic_mapping_records": [
            {"id": record.semantic_id, "labels": list(record.labels)}
            for record in validation.class_mapping.records
        ],
        "instance_mapping_records": [
            {"id": record.instance_id, "path": record.path}
            for record in validation.instance_mapping
        ],
        "instance_agreement": {
            "passed": validation.instance_agreement,
            "selected_liquid_instance_ids": list(
                validation.selected_liquid_instance_ids
            ),
            "semantic_liquid_pixels": validation.pixel_count_for("liquid"),
            "instance_liquid_pixels": validation.liquid_instance_pixel_count,
        },
        "artifacts": artifacts,
    }


def _run_capture_lifecycle(
    *,
    kit: object,
    partial: Path,
    output: Path,
    expected_frames: int,
    capture_body: Callable[[CaptureFailureContext], CaptureRun],
    stderr: object | None = None,
) -> CaptureRun:
    """Run capture, finalization, publication, and Kit shutdown in order."""

    with _diagnostic_kit_lifecycle(
        kit=kit,
        partial=partial,
        stderr=stderr,
    ) as failure_context:
        run = capture_body(failure_context)
        _finalize_capture_run(
            partial=partial,
            output=output,
            run=run,
            expected_frames=expected_frames,
            failure_context=failure_context,
        )
        return run


def _capture_to_partial(
    args: argparse.Namespace, inputs: Preflight, partial: Path
) -> CaptureRun:
    from isaacsim import SimulationApp

    width, height = inputs.resolution
    kit = SimulationApp(
        {
            "headless": True,
            "renderer": args.renderer,
            "width": width,
            "height": height,
            "extra_args": list(REPLICATOR_EXTRA_ARGS),
        }
    )
    started = time.time()

    def capture_body(failure_context: CaptureFailureContext) -> CaptureRun:
        import carb
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Sdf, Usd, UsdSemantics

        configured_pathtracing_bounces = pathtracing_bounce_settings(args)
        if configured_pathtracing_bounces:
            carb_settings = carb.settings.get_settings()
            for key, value in configured_pathtracing_bounces.items():
                carb_settings.set_int(key, value)

        failure_context.phase = "input_reverification"
        verify_input_snapshots(inputs.input_snapshots)
        failure_context.phase = "stage_open"
        context = omni.usd.get_context()
        if not context.open_stage(str(inputs.stage)):
            raise RuntimeError(f"could not open stage: {inputs.stage}")
        while context.get_stage_loading_status()[2] > 0:
            kit.update()
        verify_input_snapshots(inputs.input_snapshots)
        stage = context.get_stage()
        layer_records, layer_snapshots = _stage_layer_records(stage)
        run_input_snapshots = inputs.input_snapshots + layer_snapshots

        failure_context.phase = "stage_validation"
        start_bound = float(stage.GetStartTimeCode())
        end_bound = float(stage.GetEndTimeCode())
        outside = [
            code for code in inputs.time_codes if code < start_bound or code > end_bound
        ]
        if outside:
            raise ValueError(
                f"requested time codes outside opened stage bounds "
                f"[{start_bound}, {end_bound}]: {outside}"
            )
        time_codes_per_second = float(stage.GetTimeCodesPerSecond())
        if time_codes_per_second <= 0:
            raise RuntimeError(
                f"stage has invalid timeCodesPerSecond: {time_codes_per_second}"
            )

        camera_candidates = (
            inputs.camera_record["prim_in_csi_line"],
            inputs.camera_record["prim_in_station_file"],
        )
        camera_prim = next(
            (
                candidate
                for candidate in camera_candidates
                if stage.GetPrimAtPath(candidate).IsValid()
            ),
            None,
        )
        if camera_prim is None:
            raise RuntimeError(f"no reviewed FillCam prim found: {camera_candidates!r}")

        failure_context.phase = "semantic_setup"
        semantic_plan = _collect_semantic_plan(stage, Usd, UsdSemantics)
        transparent_annotation_plan = _collect_transparent_annotation_plan(
            stage, semantic_plan
        )
        _author_semantic_plan(stage, semantic_plan, Usd, UsdSemantics)
        failure_context.phase = "transparent_annotation_setup"
        transparent_annotation_record = _author_transparent_annotation_override(
            stage,
            transparent_annotation_plan,
            Usd,
            Sdf,
        )

        failure_context.phase = "annotator_setup"
        rep.set_global_seed(args.seed)
        render_product = rep.create.render_product(camera_prim, (width, height))
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        semantic_annotator = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={"colorize": False, "semanticTypes": ["class"]},
        )
        instance_annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        )
        for annotator in (rgb_annotator, semantic_annotator, instance_annotator):
            annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)

        failure_context.phase = "warmup"
        for _ in range(args.warmup):
            kit.update()
        verify_input_snapshots(run_input_snapshots)

        timeline = omni.timeline.get_timeline_interface()
        frame_records: list[dict[str, object]] = []
        for ordinal, time_code in enumerate(inputs.time_codes, start=1):
            failure_context.phase = "frame_capture"
            failure_context.requested_time_code = time_code
            timeline.set_current_time(time_code / time_codes_per_second)
            kit.update()
            frame_started = time.time()
            capture = _retry_shared_capture(
                step=lambda: rep.orchestrator.step(
                    rt_subframes=args.subframes,
                    delta_time=0.0,
                    wait_for_render=True,
                ),
                get_rgb=rgb_annotator.get_data,
                get_semantic=semantic_annotator.get_data,
                get_instance=instance_annotator.get_data,
                np_module=np,
                max_tries=args.max_tries,
                raster_shape=(height, width),
                require_liquid_pixels=args.require_liquid_pixels,
            )
            attempt = capture.attempts
            capture_rgb = capture.rgb
            semantic = capture.semantic
            instance = capture.instance
            last_validation = capture.validation
            failure_context.phase = "artifact_write"
            assert isinstance(capture_rgb, np.ndarray)
            assert isinstance(semantic, np.ndarray)
            assert isinstance(instance, np.ndarray)
            stem = f"{time_code:04d}"
            rgb_path = partial / "rgb" / f"{stem}.png"
            semantic_path = partial / "semantic" / f"{stem}.npy"
            instance_path = partial / "instance" / f"{stem}.npy"
            liquid_path = partial / "liquid" / f"{stem}.png"
            _write_png_checked(cv2, rgb_path, capture_rgb[..., ::-1].copy())
            _write_npy_checked(np, semantic_path, semantic)
            _write_npy_checked(np, instance_path, instance)
            liquid_ids = last_validation.class_mapping.liquid_ids
            liquid_mask = np.isin(semantic, liquid_ids).astype(np.uint8) * 255
            _write_png_checked(cv2, liquid_path, liquid_mask)

            paths = (rgb_path, semantic_path, instance_path, liquid_path)
            artifacts = {
                path.parent.name: {
                    "path": path.relative_to(partial).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in paths
            }
            frame_record = _frame_record(
                validation=last_validation,
                ordinal=ordinal,
                time_code=time_code,
                attempts=attempt,
                seconds=time.time() - frame_started,
                artifacts=artifacts,
            )
            frame_records.append(frame_record)
            print(json.dumps(frame_record, sort_keys=True), flush=True)

        failure_context.phase = "post_capture_input_reverification"
        failure_context.requested_time_code = None
        verify_input_snapshots(run_input_snapshots)
        preflight_snapshots = {
            snapshot.role: snapshot for snapshot in inputs.input_snapshots
        }
        summary: dict[str, object] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "mechanical_gates_passed": True,
            "visual_check_status": "pending",
            "inputs": {
                "twin_root": str(inputs.twin),
                "stage": {
                    "requested": args.stage,
                    "resolved_path": str(inputs.stage),
                    "sha256": preflight_snapshots["stage"].sha256,
                },
                "camera": {
                    "name": args.camera,
                    "resolved_path": str(inputs.camera_file),
                    "sha256": preflight_snapshots["camera"].sha256,
                    "prim_path": camera_prim,
                },
                "script": {
                    "resolved_path": str(inputs.script),
                    "sha256": preflight_snapshots["script"].sha256,
                },
                "non_session_usd_layer_stack": layer_records,
            },
            "render_settings": {
                "headless": True,
                "renderer": args.renderer,
                **pathtracing_bounce_summary(args),
                "resolution": [width, height],
                "frame_spec": args.frames,
                "time_codes": list(inputs.time_codes),
                "time_codes_per_second": time_codes_per_second,
                "stage_time_code_bounds": [start_bound, end_bound],
                "warmup_updates": args.warmup,
                "rt_subframes": args.subframes,
                "max_tries_per_frame": args.max_tries,
                "capture_on_play": False,
                "delta_time": 0.0,
                "wait_for_render": True,
                "annotators": {
                    "rgb": {},
                    "semantic_segmentation": {
                        "colorize": False,
                        "semanticTypes": ["class"],
                    },
                    "instance_segmentation": {"colorize": False},
                },
                "require_liquid_pixels": args.require_liquid_pixels,
                "replicator_global_seed": args.seed,
                "seed_limit": "The recorded Replicator seed does not prove full renderer determinism.",
            },
            "semantic_override": asdict(LIQUID_OVERRIDE),
            "transparent_annotation_override": transparent_annotation_record,
            "semantic_plan": {
                "copied_assignments": [
                    asdict(assignment) for assignment in semantic_plan.copied_assignments
                ],
                "liquid_assignment": asdict(semantic_plan.liquid_assignment),
                "excluded_subtree": list(semantic_plan.excluded_subtree),
            },
            "frame_count": len(frame_records),
            "total_seconds": round(time.time() - started, 6),
            "frames": frame_records,
        }
        failure_context.phase = "summary_write"
        summary_path = partial / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise OSError(f"summary write verification failed: {summary_path}")
        run = CaptureRun(summary=summary, input_snapshots=run_input_snapshots)
        return run

    return _run_capture_lifecycle(
        kit=kit,
        partial=partial,
        output=inputs.output,
        expected_frames=len(inputs.time_codes),
        capture_body=capture_body,
    )


def _validate_complete_partial(
    partial: Path, summary: dict[str, object], expected_frames: int
) -> None:
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise RuntimeError("summary schema version failed")
    if summary.get("mechanical_gates_passed") is not True:
        raise RuntimeError("summary mechanical gate failed")
    if summary.get("visual_check_status") != "pending":
        raise RuntimeError("summary visual status must remain pending")
    frames = summary.get("frames")
    if not isinstance(frames, list) or len(frames) != expected_frames:
        raise RuntimeError("summary frame denominator failed")
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get("artifacts"), dict):
            raise RuntimeError("summary artifact records are malformed")
        for artifact in frame["artifacts"].values():
            if not isinstance(artifact, dict):
                raise RuntimeError("summary artifact record is malformed")
            relative = artifact.get("path")
            expected_hash = artifact.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise RuntimeError("summary artifact path/hash is malformed")
            path = partial / relative
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise RuntimeError(f"artifact hash gate failed: {relative}")
    summary_path = partial / "summary.json"
    if not summary_path.is_file() or json.loads(
        summary_path.read_text(encoding="utf-8")
    ) != summary:
        raise RuntimeError("summary file gate failed")


def _finalize_capture_run(
    *,
    partial: Path,
    output: Path,
    run: CaptureRun,
    expected_frames: int,
    failure_context: CaptureFailureContext,
) -> None:
    """Validate, reverify, then publish as the final fallible capture action."""

    failure_context.phase = "complete_partial_validation"
    _validate_complete_partial(partial, run.summary, expected_frames)
    failure_context.phase = "final_input_reverification"
    verify_input_snapshots(run.input_snapshots)
    failure_context.phase = "publication"
    publish_partial_no_clobber(partial, output)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = preflight(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        build_parser().error(str(error))
    partial = _create_partial_output(inputs.output)
    _capture_to_partial(args, inputs, partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

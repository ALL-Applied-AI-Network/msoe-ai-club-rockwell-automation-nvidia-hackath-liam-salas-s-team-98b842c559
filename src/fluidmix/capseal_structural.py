"""CPU-only contracts for a paired CapSealCam structural render pilot.

Nothing in this module imports Isaac Sim.  It validates render inputs, produces the
deterministic lens-warp contracts, and evaluates already captured arrays.  Passing these
checks does not demonstrate that Isaac authored the requested scene state or that a
synthetic signal transfers to a physical cap/seal station.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from .render_capture import (
    INSTANCE_BACKGROUND_ID,
    resolve_class_ids,
    resolve_instance_paths,
)


OVERSCAN_RASTER_SIZE = (768, 576)
FINAL_RASTER_SIZE = (640, 480)
TARGETS = frozenset(f"G{number:02d}" for number in range(1, 31))
STATES = ("intact", "missing", "unseated")
UNSEATED_Z_LOCAL_UNITS = 0.4
UNSEATED_X_DEGREES = 10.0
UNSEATED_DISPLACEMENT_MM = 10.16


class StateValidationError(ValueError):
    """The requested target or structural state is outside this bounded pilot."""


class OutputPreflightError(ValueError):
    """The requested immutable output destination is unsuitable."""


class GateError(ValueError):
    """One or more capture acceptance gates did not pass."""


@dataclass(frozen=True)
class SourcePose:
    """Frozen source root attributes, in the authored USD local coordinate system."""

    visibility: str
    translation: tuple[float, float, float]
    rotation_xyz_degrees: tuple[float, float, float]


@dataclass(frozen=True)
class StructuralState:
    """One state derived directly from :class:`SourcePose`, never a prior state."""

    visibility: str
    translation: tuple[float, float, float]
    rotation_xyz_degrees: tuple[float, float, float]
    displacement_mm: float


@dataclass(frozen=True)
class InputSnapshot:
    role: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class TargetMask:
    """A mask resolved only from an exact structured annotator mapping."""

    mask: np.ndarray
    selected_ids: tuple[int, ...]


def validate_cap_pose_layout(
    op_names: Sequence[object], op_type_names: Sequence[object]
) -> tuple[str, str, str]:
    """Permit only the reviewed authored translate/rotateXYZ/scale op stack."""

    expected_names = ("xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale")
    expected_types = ("translate", "rotateXYZ", "scale")
    if tuple(op_names) != expected_names or tuple(op_type_names) != expected_types:
        raise ValueError("cap root xform op layout must be translate, rotateXYZ, scale")
    return expected_names


def _integer_id_raster(value: object, kind: str) -> np.ndarray:
    raster = np.asarray(value)
    if raster.ndim != 2 or raster.dtype.kind not in {"u", "i"}:
        raise ValueError(f"{kind} raster must be two-dimensional integers")
    if np.any(raster < 0):
        raise ValueError(f"{kind} raster has negative IDs")
    return raster


def target_semantic_mask(
    semantic: object,
    id_to_labels: object,
    class_name: str,
    *,
    expected_visible: bool,
) -> TargetMask:
    """Resolve a target class exactly; a hidden state may truthfully omit it."""

    raster = _integer_id_raster(semantic, "semantic")
    if type(expected_visible) is not bool:
        raise ValueError("expected semantic visibility must be bool")
    mapping = resolve_class_ids(id_to_labels)
    visible_ids = {int(value) for value in np.unique(raster)}
    mapped_ids = {record.semantic_id for record in mapping.records}
    unmapped = sorted(visible_ids.difference(mapped_ids))
    if unmapped:
        raise ValueError(f"semantic raster has unmapped IDs: {unmapped!r}")
    selected_ids = mapping.ids_for(class_name)
    mask = np.isin(raster, selected_ids)
    if expected_visible and not selected_ids:
        raise ValueError(f"semantic class absent: {class_name}")
    if expected_visible and not np.any(mask):
        raise ValueError(f"semantic class pixels empty: {class_name}")
    if not expected_visible and np.any(mask):
        raise ValueError(f"hidden semantic class has pixels: {class_name}")
    return TargetMask(mask.astype(bool), selected_ids)


def target_instance_mask(
    instance: object,
    id_to_paths: object,
    leaf_path: str,
    *,
    expected_visible: bool,
) -> TargetMask:
    """Resolve one target instance by exact leaf path, never a text substring."""

    raster = _integer_id_raster(instance, "instance")
    if type(expected_visible) is not bool:
        raise ValueError("expected instance visibility must be bool")
    if not isinstance(leaf_path, str) or not leaf_path.startswith("/") or leaf_path.endswith("/"):
        raise ValueError("instance leaf path must be absolute")
    records = resolve_instance_paths(id_to_paths)
    path_by_id = {record.instance_id: record.path for record in records}
    visible_ids = {int(value) for value in np.unique(raster)}
    unmapped = sorted(visible_ids.difference(path_by_id).difference({INSTANCE_BACKGROUND_ID}))
    if unmapped:
        raise ValueError(f"instance raster has unmapped IDs: {unmapped!r}")
    selected_ids = tuple(
        sorted(
            instance_id
            for instance_id, path in path_by_id.items()
            if path == leaf_path and instance_id in visible_ids
        )
    )
    mask = np.isin(raster, selected_ids)
    if expected_visible and not selected_ids:
        raise ValueError(f"instance path absent: {leaf_path}")
    if expected_visible and not np.any(mask):
        raise ValueError(f"instance path pixels empty: {leaf_path}")
    if not expected_visible and np.any(mask):
        raise ValueError(f"hidden instance path has pixels: {leaf_path}")
    return TargetMask(mask.astype(bool), selected_ids)


def validate_target(target: object) -> str:
    """Require exactly one tray identifier from ``G01`` through ``G30``."""

    if not isinstance(target, str) or target not in TARGETS:
        raise StateValidationError("target must be exactly G01 through G30")
    return target


def validate_states(states: object) -> tuple[str, ...]:
    """Parse a comma-delimited, duplicate-free nonempty state list."""

    if not isinstance(states, str) or not states:
        raise StateValidationError("states must be a nonempty comma-delimited string")
    values = tuple(states.split(","))
    if any(not value or value not in STATES for value in values):
        raise StateValidationError("states contain an unknown or empty state")
    if len(set(values)) != len(values):
        raise StateValidationError("states contain a duplicate state")
    return values


def _float3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise StateValidationError(f"{name} must contain exactly three numeric values")
    converted = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in converted):
        raise StateValidationError(f"{name} must be finite")
    return converted  # type: ignore[return-value]


def apply_structural_state(source: SourcePose, state: str) -> StructuralState:
    """Derive one state from the source values, avoiding accumulated mutations."""

    if not isinstance(source, SourcePose):
        raise StateValidationError("source pose is malformed")
    if state not in STATES:
        raise StateValidationError("state is unknown")
    translation = _float3(source.translation, "source translation")
    rotation = _float3(source.rotation_xyz_degrees, "source rotation")
    if not isinstance(source.visibility, str) or not source.visibility:
        raise StateValidationError("source visibility is malformed")
    if state == "intact":
        return StructuralState(source.visibility, translation, rotation, 0.0)
    if state == "missing":
        return StructuralState("invisible", translation, rotation, 0.0)
    return StructuralState(
        "inherited",
        (translation[0], translation[1], translation[2] + UNSEATED_Z_LOCAL_UNITS),
        (rotation[0] + UNSEATED_X_DEGREES, rotation[1], rotation[2]),
        UNSEATED_DISPLACEMENT_MM,
    )


def central_crop_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Return identity-lens sample coordinates for the centered 768x576 overscan crop."""

    overscan_width, overscan_height = OVERSCAN_RASTER_SIZE
    final_width, final_height = FINAL_RASTER_SIZE
    left = (overscan_width - final_width) // 2
    top = (overscan_height - final_height) // 2
    source_x, source_y = np.meshgrid(
        np.arange(left, left + final_width, dtype=np.float64),
        np.arange(top, top + final_height, dtype=np.float64),
    )
    return source_x, source_y


def _validate_warp_coordinates(source_x: object, source_y: object) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(source_x, dtype=np.float64)
    y = np.asarray(source_y, dtype=np.float64)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError("lens coordinate arrays must be two-dimensional and share a shape")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("lens coordinates must be finite")
    return x, y


def warp_rgb_linear(rgb: object, source_x: object, source_y: object) -> np.ndarray:
    """Lens-warp RGB with bilinear interpolation and black pixels outside the source."""

    raster = np.asarray(rgb)
    if raster.ndim != 3 or raster.shape[2] != 3:
        raise ValueError("RGB raster must have shape (height, width, 3)")
    if raster.dtype.kind not in {"u", "i", "f"}:
        raise ValueError("RGB raster must be numeric")
    x, y = _validate_warp_coordinates(source_x, source_y)
    height, width = raster.shape[:2]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x0 = np.clip(x0, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (x - np.floor(x))[..., None]
    dy = (y - np.floor(y))[..., None]
    upper = raster[y0, x0] * (1.0 - dx) + raster[y0, x1] * dx
    lower = raster[y1, x0] * (1.0 - dx) + raster[y1, x1] * dx
    result = upper * (1.0 - dy) + lower * dy
    result[~valid] = 0
    if raster.dtype.kind in {"u", "i"}:
        return np.rint(result).astype(raster.dtype)
    return result.astype(raster.dtype)


def warp_ids_nearest(ids: object, source_x: object, source_y: object) -> np.ndarray:
    """Lens-warp integer IDs exactly, with ID 0 for coordinates outside the input."""

    raster = np.asarray(ids)
    if raster.ndim != 2 or raster.dtype.kind not in {"u", "i"}:
        raise ValueError("ID raster must be a two-dimensional integer array")
    x, y = _validate_warp_coordinates(source_x, source_y)
    nearest_x = np.floor(x + 0.5).astype(np.intp)
    nearest_y = np.floor(y + 0.5).astype(np.intp)
    height, width = raster.shape
    valid = (
        (nearest_x >= 0)
        & (nearest_x < width)
        & (nearest_y >= 0)
        & (nearest_y < height)
    )
    result = np.zeros(x.shape, dtype=raster.dtype)
    result[valid] = raster[nearest_y[valid], nearest_x[valid]]
    return result


def preflight_output_path(output: str | Path) -> Path:
    """Resolve an as-yet unoccupied output directory without creating it."""

    candidate = Path(output)
    if candidate.exists() or candidate.is_symlink():
        raise OutputPreflightError(f"output path is occupied: {candidate}")
    parent = candidate.parent
    if not parent.exists() or not parent.is_dir() or parent.is_symlink():
        raise OutputPreflightError(f"output parent must be an existing directory: {parent}")
    return candidate.resolve()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_input_file(role: str, path: str | Path) -> InputSnapshot:
    """Hash one stable, regular input file and reject a concurrent replacement."""

    candidate = Path(path).resolve(strict=True)
    before = candidate.stat()
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"input {role} must be a regular non-symlink file: {candidate}")
    digest = sha256_file(candidate)
    after = candidate.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError(f"input {role} changed while hashing: {candidate}")
    return InputSnapshot(role, candidate, digest, after.st_size)


def verify_input_snapshots(snapshots: Sequence[InputSnapshot]) -> None:
    for snapshot in snapshots:
        actual = snapshot_input_file(snapshot.role, snapshot.path)
        if actual != snapshot:
            raise ValueError(f"input {snapshot.role} drifted after its snapshot")


def validate_camera_record(record: object) -> dict[str, float | int]:
    """Validate the fixed overscan and the lens fields needed for CapSealCam."""

    if not isinstance(record, Mapping):
        raise ValueError("camera JSON record must be an object")

    def field(*names: str) -> object:
        for name in names:
            if name in record:
                return record[name]
        raise ValueError(f"camera JSON is missing {names[0]}")

    if record.get("camera") != "CapSealCam":
        raise ValueError("camera JSON must identify CapSealCam")
    render_px = record.get("render_px")
    if not isinstance(render_px, Sequence) or isinstance(render_px, (str, bytes)) or len(render_px) != 2:
        raise ValueError("camera render_px must contain width and height")
    width, height = render_px
    if type(width) is not int or type(height) is not int:
        raise ValueError("camera dimensions must be integer pixels")
    if (width, height) != OVERSCAN_RASTER_SIZE:
        raise ValueError("camera dimensions must be the supplied 768 x 576 overscan")
    lens: dict[str, float | int] = {"width": width, "height": height}
    final_px = record.get("final_image_px")
    if final_px != list(FINAL_RASTER_SIZE):
        raise ValueError("camera final_image_px must be [640, 480]")
    for canonical, positive in (("focal_px_final", True), ("lens_k1", False)):
        value = field(canonical)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or (positive and value <= 0):
            qualifier = "positive finite" if positive else "finite"
            raise ValueError(f"camera {canonical} must be a {qualifier} number")
        lens[canonical] = float(value)
    return lens


def _binary_mask(value: object, name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != FINAL_RASTER_SIZE[::-1]:
        raise GateError(f"{name} must have shape (480, 640)")
    if mask.dtype == np.bool_:
        return mask
    if mask.dtype.kind in {"u", "i"} and np.all((mask == 0) | (mask == 1)):
        return mask.astype(bool)
    raise GateError(f"{name} must be a binary mask")


def mask_statistics(mask: object) -> dict[str, object]:
    """Count a binary mask and report an inclusive ``[xmin, ymin, xmax, ymax]`` box."""

    binary = _binary_mask(mask, "mask")
    ys, xs = np.nonzero(binary)
    if not len(xs):
        return {"count": 0, "bbox_xyxy_inclusive": None, "centroid_xy": None}
    return {
        "count": int(len(xs)),
        "bbox_xyxy_inclusive": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
    }


def _state_gate_facts(name: str, artifacts: Mapping[str, object]) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    required = {
        "rgb",
        "semantic",
        "instance",
        "target_cap_semantic_mask",
        "target_cap_instance_mask",
        "target_vial_mask",
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise GateError(f"{name} artifacts are missing: {', '.join(missing)}")
    rgb = np.asarray(artifacts["rgb"])
    semantic = np.asarray(artifacts["semantic"])
    instance = np.asarray(artifacts["instance"])
    cap_semantic = _binary_mask(artifacts["target_cap_semantic_mask"], f"{name} cap semantic mask")
    cap_instance = _binary_mask(artifacts["target_cap_instance_mask"], f"{name} cap instance mask")
    vial = _binary_mask(artifacts["target_vial_mask"], f"{name} vial mask")
    rgb_ok = rgb.shape == (480, 640, 3) and rgb.dtype.kind in {"u", "i", "f"} and bool(np.ptp(rgb) > 0)
    aovs_ok = (
        semantic.shape == (480, 640)
        and instance.shape == (480, 640)
        and semantic.dtype.kind in {"u", "i"}
        and instance.dtype.kind in {"u", "i"}
    )
    facts = {
        "rgb_readable_640x480_nonconstant": {"passed": rgb_ok, "shape": list(rgb.shape), "dtype": str(rgb.dtype)},
        "aovs_readable_640x480": {"passed": aovs_ok, "semantic_shape": list(semantic.shape), "instance_shape": list(instance.shape)},
        "target_cap_semantic_instance_agreement": {"passed": bool(np.array_equal(cap_semantic, cap_instance))},
        "target_cap": mask_statistics(cap_semantic),
        "target_vial": mask_statistics(vial),
    }
    return facts, cap_semantic, vial


def evaluate_structural_gates(
    artifacts_by_state: Mapping[str, Mapping[str, object]], *, require_pass: bool = False
) -> dict[str, object]:
    """Evaluate every mechanical gate over the required three paired state captures."""

    if set(artifacts_by_state) != set(STATES):
        raise GateError("gate evaluation requires exactly intact, missing, and unseated artifacts")
    states: dict[str, object] = {}
    cap_masks: dict[str, np.ndarray] = {}
    vial_masks: dict[str, np.ndarray] = {}
    failed: list[str] = []
    for state in STATES:
        facts, cap, vial = _state_gate_facts(state, artifacts_by_state[state])
        states[state] = facts
        cap_masks[state] = cap
        vial_masks[state] = vial
        for gate_name in ("rgb_readable_640x480_nonconstant", "aovs_readable_640x480", "target_cap_semantic_instance_agreement"):
            if not facts[gate_name]["passed"]:  # type: ignore[index]
                failed.append(f"{state}.{gate_name}")
        cap_count = int(facts["target_cap"]["count"])  # type: ignore[index]
        vial_count = int(facts["target_vial"]["count"])  # type: ignore[index]
        if vial_count == 0:
            failed.append(f"{state}.target_vial_nonempty")
        expected_cap_visible = state != "missing"
        if (cap_count > 0) != expected_cap_visible:
            failed.append(f"{state}.target_cap_visibility")
    intact_vial = vial_masks["intact"]
    missing_vial = vial_masks["missing"]
    intact_only_count = int(np.count_nonzero(intact_vial & ~missing_vial))
    newly_visible_count = int(np.count_nonzero(missing_vial & ~intact_vial))
    vial_subset = intact_only_count == 0
    paired = {
        "intact_vial_visible_subset_of_missing": {
            "passed": vial_subset,
            "intact_only_pixels": intact_only_count,
            "newly_visible_missing_state_pixels": newly_visible_count,
        },
        "intact_missing_target_vial_exact_match_diagnostic": {
            "identical": bool(np.array_equal(intact_vial, missing_vial)),
            "hard_gate": False,
        },
    }
    if not vial_subset:
        failed.append("intact_vial_visible_subset_of_missing")
    intact_centroid = mask_statistics(cap_masks["intact"])["centroid_xy"]
    unseated_centroid = mask_statistics(cap_masks["unseated"])["centroid_xy"]
    distance: float | None = None
    if intact_centroid is not None and unseated_centroid is not None:
        distance = float(np.hypot(unseated_centroid[0] - intact_centroid[0], unseated_centroid[1] - intact_centroid[1]))
    centroid_ok = distance is not None and 2.0 <= distance <= 20.0
    paired["unseated_cap_centroid_shift_2_to_20_pixels"] = {"passed": centroid_ok, "distance_pixels": distance}
    if not centroid_ok:
        failed.append("unseated_cap_centroid_shift_2_to_20_pixels")
    result: dict[str, object] = {"passed": not failed, "failed": failed, "states": states, "paired": paired}
    if require_pass and failed:
        raise GateError("structural gates failed: " + ", ".join(failed))
    return result


def build_structural_receipt(
    *,
    target: str,
    states: Sequence[str],
    input_hashes: Mapping[str, str],
    gates: Mapping[str, object],
) -> dict[str, object]:
    """Build the evidence-bounded receipt shared by the runtime publisher and CPU tests."""

    return {
        "receipt_type": "capseal_structural_pilot",
        "target": validate_target(target),
        "states": list(validate_states(",".join(states))),
        "input_sha256": dict(input_hashes),
        "gates": dict(gates),
        "real_data_used": False,
        "claim_boundary": "synthetic structural pilot only; no real-data claim",
        "unseated_displacement": {
            "local_z_units": UNSEATED_Z_LOCAL_UNITS,
            "millimetres": UNSEATED_DISPLACEMENT_MM,
            "calibration": "uncalibrated synthetic displacement",
        },
    }


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "atomic renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(number, os.strerror(number), os.fspath(destination))
    raise OSError(number, os.strerror(number), os.fspath(destination))


def publish_directory_no_clobber(output: str | Path, populate: Callable[[Path], None]) -> Path:
    """Populate a sibling temporary directory, then atomically publish it once."""

    destination = preflight_output_path(output)
    partial = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        populate(partial)
        _atomic_rename_noreplace(partial, destination)
        return destination
    finally:
        if partial.exists():
            shutil.rmtree(partial)

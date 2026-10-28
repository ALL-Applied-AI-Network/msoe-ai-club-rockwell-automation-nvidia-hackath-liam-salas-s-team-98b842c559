"""Pure, fail-closed contracts for the no-physics rigid vial transport."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = "/FillStation/FillStation_LocalRoot"
SOURCE_LAYER_HASHES = {
    "fill_station.usda": "5f58540c187ac16c4426bffb3ebbe559e62d6d63ff359afdb08867fc24b60af9",
    "geometry/fill_station.usdc": (
        "57f578fd047d3a5f2ee25712d841f836272fc745a89c5dfbcc301f0c199191db"
    ),
}
SOURCE_LAYER_BYTES = {
    "fill_station.usda": 906,
    "geometry/fill_station.usdc": 26_464_568,
}

OUTGOING_PRIM_PATHS = frozenset(
    {
        f"{ROOT}/Fill_vial_active",
        f"{ROOT}/FluidmixF2CountDrivenLiquid",
        f"{ROOT}/Fill_barcode_label",
        f"{ROOT}/Fill_label_top_rule",
        f"{ROOT}/Fill_barcode_00",
        f"{ROOT}/Fill_barcode_01",
        f"{ROOT}/Fill_barcode_02",
        f"{ROOT}/Fill_barcode_03",
        f"{ROOT}/Fill_barcode_04",
        f"{ROOT}/Fill_barcode_05",
        f"{ROOT}/Fill_barcode_06",
        f"{ROOT}/Fill_barcode_07",
        f"{ROOT}/Fill_barcode_08",
        f"{ROOT}/Fill_barcode_09",
        f"{ROOT}/Fill_barcode_10",
        f"{ROOT}/Fill_barcode_11",
        f"{ROOT}/Fill_barcode_12",
        f"{ROOT}/Fill_barcode_13",
        f"{ROOT}/Fill_barcode_14",
        f"{ROOT}/Fill_barcode_15",
        f"{ROOT}/Fill_barcode_16",
        f"{ROOT}/Fill_barcode_17",
        f"{ROOT}/Fill_barcode_18",
        f"{ROOT}/Fill_barcode_19",
        f"{ROOT}/Fill_barcode_20",
        f"{ROOT}/Fill_barcode_21",
        f"{ROOT}/Fill_barcode_22",
        f"{ROOT}/Fill_barcode_23",
        f"{ROOT}/Fill_barcode_24",
        f"{ROOT}/Fill_barcode_25",
        f"{ROOT}/Fill_barcode_26",
        f"{ROOT}/V3_carrier_0_cart_00",
        f"{ROOT}/V3_carrier_0_cart_01",
        f"{ROOT}/V3_carrier_0_cart_02",
        f"{ROOT}/V3_carrier_0_cart_03",
        f"{ROOT}/V3_carrier_0_cart_04",
        f"{ROOT}/V3_carrier_0_cart_05",
        f"{ROOT}/V3_carrier_0_cart_06",
        f"{ROOT}/V3_carrier_0_cart_07",
        f"{ROOT}/V3_carrier_0_cart_08",
        f"{ROOT}/V3_carrier_0_cart_09",
        f"{ROOT}/V3_carrier_0_cart_10",
        f"{ROOT}/V3_carrier_0_cart_11",
        f"{ROOT}/V3_carrier_0_cart_12",
        f"{ROOT}/V3_carrier_0_cart_13",
        f"{ROOT}/V3_carrier_0_cart_14",
        f"{ROOT}/V3_carrier_0_cart_15",
        f"{ROOT}/V3_carrier_0_cart_16",
        f"{ROOT}/V3_carrier_0_base_19",
        f"{ROOT}/V3_carrier_0_post_20",
        f"{ROOT}/V3_carrier_0_clamp_21",
        f"{ROOT}/V3_carrier_0_setscrew_17",
        f"{ROOT}/V3_carrier_0_setscrew_18",
        f"{ROOT}/V3_estimated_fastener_0",
        f"{ROOT}/V3_estimated_fastener_0_001",
        f"{ROOT}/V3_estimated_fastener_0_002",
        f"{ROOT}/V3_estimated_fastener_0_003",
    }
)

INCOMING_PRIM_PATHS = frozenset(
    {
        f"{ROOT}/Fill_vial_neighbour_1",
        f"{ROOT}/V3_carrier_1_cart_00",
        f"{ROOT}/V3_carrier_1_cart_01",
        f"{ROOT}/V3_carrier_1_cart_02",
        f"{ROOT}/V3_carrier_1_cart_03",
        f"{ROOT}/V3_carrier_1_cart_04",
        f"{ROOT}/V3_carrier_1_cart_05",
        f"{ROOT}/V3_carrier_1_cart_06",
        f"{ROOT}/V3_carrier_1_cart_07",
        f"{ROOT}/V3_carrier_1_cart_08",
        f"{ROOT}/V3_carrier_1_cart_09",
        f"{ROOT}/V3_carrier_1_cart_10",
        f"{ROOT}/V3_carrier_1_cart_11",
        f"{ROOT}/V3_carrier_1_cart_12",
        f"{ROOT}/V3_carrier_1_cart_13",
        f"{ROOT}/V3_carrier_1_cart_14",
        f"{ROOT}/V3_carrier_1_cart_15",
        f"{ROOT}/V3_carrier_1_cart_16",
        f"{ROOT}/V3_carrier_1_base_19",
        f"{ROOT}/V3_carrier_1_post_20",
        f"{ROOT}/V3_carrier_1_clamp_21",
        f"{ROOT}/V3_carrier_1_setscrew_17",
        f"{ROOT}/V3_carrier_1_setscrew_18",
        f"{ROOT}/V3_estimated_fastener_1",
        f"{ROOT}/V3_estimated_fastener_1_001",
        f"{ROOT}/V3_estimated_fastener_1_002",
        f"{ROOT}/V3_estimated_fastener_1_003",
    }
)

QUEUED_PRIM_PATHS = frozenset(
    {
        f"{ROOT}/Fill_vial_neighbour_2",
        f"{ROOT}/V3_carrier_2_cart_00",
        f"{ROOT}/V3_carrier_2_cart_01",
        f"{ROOT}/V3_carrier_2_cart_02",
        f"{ROOT}/V3_carrier_2_cart_03",
        f"{ROOT}/V3_carrier_2_cart_04",
        f"{ROOT}/V3_carrier_2_cart_05",
        f"{ROOT}/V3_carrier_2_cart_06",
        f"{ROOT}/V3_carrier_2_cart_07",
        f"{ROOT}/V3_carrier_2_cart_08",
        f"{ROOT}/V3_carrier_2_cart_09",
        f"{ROOT}/V3_carrier_2_cart_10",
        f"{ROOT}/V3_carrier_2_cart_11",
        f"{ROOT}/V3_carrier_2_cart_12",
        f"{ROOT}/V3_carrier_2_cart_13",
        f"{ROOT}/V3_carrier_2_cart_14",
        f"{ROOT}/V3_carrier_2_cart_15",
        f"{ROOT}/V3_carrier_2_cart_16",
        f"{ROOT}/V3_carrier_2_base_19",
        f"{ROOT}/V3_carrier_2_post_20",
        f"{ROOT}/V3_carrier_2_clamp_21",
        f"{ROOT}/V3_carrier_2_setscrew_17",
        f"{ROOT}/V3_carrier_2_setscrew_18",
        f"{ROOT}/V3_estimated_fastener_2",
        f"{ROOT}/V3_estimated_fastener_2_001",
        f"{ROOT}/V3_estimated_fastener_2_002",
        f"{ROOT}/V3_estimated_fastener_2_003",
    }
)

TRANSPORT_GROUPS = {
    "outgoing": tuple(sorted(OUTGOING_PRIM_PATHS)),
    "incoming": tuple(sorted(INCOMING_PRIM_PATHS)),
    "queued": tuple(sorted(QUEUED_PRIM_PATHS)),
}
_MANAGED_PREFIXES = (
    f"{ROOT}/Fill_vial_active",
    f"{ROOT}/Fill_vial_neighbour_",
    f"{ROOT}/Fill_vial_neighbor_",
    f"{ROOT}/Fill_barcode",
    f"{ROOT}/Fill_label_",
    f"{ROOT}/FluidmixF2CountDrivenLiquid",
    f"{ROOT}/V3_carrier_",
    f"{ROOT}/V3_estimated_fastener_",
)
_EXPECTED_PRIM_PATHS = frozenset().union(*TRANSPORT_GROUPS.values())
_SNAPSHOT_ARRAY_NAMES = frozenset(
    {
        "simulation_points",
        "positions",
        "velocities",
        "orientations",
        "scales",
        "ids",
        "proto_indices",
        "scheduled_sources",
        "birth_compositions",
        "current_compositions",
        "birth_steps",
        "birth_event_ids",
        "pour_event_ids",
    }
)

ACTIVE_VIAL_PATH = f"{ROOT}/Fill_vial_active"
INCOMING_VIAL_PATH = f"{ROOT}/Fill_vial_neighbour_1"
QUEUED_VIAL_PATH = f"{ROOT}/Fill_vial_neighbour_2"
INCOMING_FINAL_DELTA_Y_M = 0.0848413319824885
OUTGOING_INITIAL_BOUND_Y_M = 0.19
COUNT_BODY_PATH = f"{ROOT}/FluidmixF2CountDrivenLiquid"
COUNT_BODY_MESH_PATH = f"{COUNT_BODY_PATH}/Fill_liquid"
COUNT_BODY_RECONSTRUCTION_PROPERTY_PATHS = frozenset(
    {
        f"{COUNT_BODY_PATH}.xformOp:transform",
        f"{COUNT_BODY_MESH_PATH}.doubleSided",
        f"{COUNT_BODY_MESH_PATH}.extent",
        f"{COUNT_BODY_MESH_PATH}.faceVaryingLinearInterpolation",
        f"{COUNT_BODY_MESH_PATH}.faceVertexCounts",
        f"{COUNT_BODY_MESH_PATH}.faceVertexIndices",
        f"{COUNT_BODY_MESH_PATH}.interpolateBoundary",
        f"{COUNT_BODY_MESH_PATH}.material:binding",
        f"{COUNT_BODY_MESH_PATH}.orientation",
        f"{COUNT_BODY_MESH_PATH}.points",
        f"{COUNT_BODY_MESH_PATH}.primvars:displayColor",
        f"{COUNT_BODY_MESH_PATH}.semantics:labels:class",
        f"{COUNT_BODY_MESH_PATH}.subdivisionScheme",
        f"{COUNT_BODY_MESH_PATH}.triangleSubdivisionRule",
        f"{COUNT_BODY_MESH_PATH}.visibility",
        f"{COUNT_BODY_MESH_PATH}.xformOp:transform",
        f"{COUNT_BODY_MESH_PATH}.xformOpOrder",
    }
)
TRANSPORT_XFORM_OP = "xformOp:translate:fluidmixTransport"
COUNT_BODY_RECONSTRUCTION_PRIM_PATHS = (COUNT_BODY_PATH, COUNT_BODY_MESH_PATH)
SESSION_PRIM_PATHS = tuple(
    sorted({"/FillStation", ROOT, *_EXPECTED_PRIM_PATHS, COUNT_BODY_MESH_PATH})
)
TRANSPORT_PROPERTY_PATHS = frozenset(
    property_path
    for path in _EXPECTED_PRIM_PATHS
    for property_path in (f"{path}.{TRANSPORT_XFORM_OP}", f"{path}.xformOpOrder")
)
SOURCE_XFORM_OP_ORDER = (
    "xformOp:translate",
    "xformOp:rotateXYZ",
    "xformOp:scale",
)


@dataclass(frozen=True)
class RigidTransportSpec:
    """Frozen temporal and world-space contract for the transport segment."""

    frame_count: int = 21
    fps: float = 10.0
    outgoing_final_delta_y_m: float = OUTGOING_INITIAL_BOUND_Y_M
    incoming_final_delta_y_m: float = INCOMING_FINAL_DELTA_Y_M

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_count, bool)
            or not isinstance(self.frame_count, int)
            or self.frame_count != 21
        ):
            raise ValueError("frame_count must be exactly 21")
        if (
            type(self.fps) is not float
            or not math.isfinite(self.fps)
            or self.fps != 10.0
        ):
            raise ValueError("fps must be exactly 10.0")
        if (
            type(self.outgoing_final_delta_y_m) is not float
            or not math.isfinite(self.outgoing_final_delta_y_m)
            or self.outgoing_final_delta_y_m < OUTGOING_INITIAL_BOUND_Y_M
        ):
            raise ValueError("outgoing_final_delta_y_m must be at least +0.19 m")
        if (
            type(self.incoming_final_delta_y_m) is not float
            or not math.isfinite(self.incoming_final_delta_y_m)
            or self.incoming_final_delta_y_m != INCOMING_FINAL_DELTA_Y_M
        ):
            raise ValueError(
                "incoming_final_delta_y_m must equal +0.0848413319824885 m"
            )


def resolve_transport_groups(
    prim_paths: Collection[str], source_layer_hashes: Mapping[str, str]
) -> Mapping[str, tuple[str, ...]]:
    """Bind exact reviewed mover paths to the two immutable source layers."""

    if not isinstance(source_layer_hashes, Mapping):
        raise TypeError("source_layer_hashes must be a mapping")
    if dict(source_layer_hashes) != SOURCE_LAYER_HASHES:
        raise ValueError("source layer binding differs from the reviewed hashes")
    if isinstance(prim_paths, (str, bytes)) or not isinstance(prim_paths, Collection):
        raise TypeError("prim_paths must be a collection of absolute prim paths")
    paths = set(prim_paths)
    if len(paths) != len(prim_paths) or any(
        not isinstance(path, str) or not path.startswith("/") for path in paths
    ):
        raise ValueError("prim inventory contains duplicate or invalid paths")
    managed_paths = {
        path for path in paths if any(path.startswith(prefix) for prefix in _MANAGED_PREFIXES)
    }
    if managed_paths != _EXPECTED_PRIM_PATHS:
        raise ValueError("prim inventory differs from the reviewed transport prim inventory")
    return dict(TRANSPORT_GROUPS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{label} contains non-finite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc


def _artifact_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or str(relative) in {"", "."}:
        raise ValueError(f"{label} path must be a safe relative path")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{label} path escapes the accepted case")
    if not path.is_file():
        raise ValueError(f"{label} path is missing")
    return path


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_layer_key(value: str) -> str | None:
    path = Path(value)
    if path.name == "fill_station.usda":
        return "fill_station.usda"
    if path.name == "fill_station.usdc" and path.parent.name == "geometry":
        return "geometry/fill_station.usdc"
    return None


def load_rigid_transport_inputs(case_output: Path, stage_path: Path) -> dict[str, object]:
    """Load one accepted case only after every provenance byte has been checked.

    This function returns provenance and simulation arrays for Part A rendering.  It
    does not construct model-input records; the transport receipt and arrays remain
    excluded from Part B.
    """

    case_root = Path(case_output).resolve()
    source_stage = Path(stage_path).resolve()
    if not case_root.is_dir():
        raise ValueError("accepted case output is missing")
    if not source_stage.is_file():
        raise ValueError("source stage is missing")

    receipt_path = _artifact_path(case_root, "receipt.json", "accepted receipt")
    receipt_value = _read_json(receipt_path, "accepted receipt")
    if not isinstance(receipt_value, Mapping):
        raise ValueError("accepted receipt must be a JSON object")
    receipt = dict(receipt_value)
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 20
        or receipt.get("experiment") != "physx_three_pour_f2_paint_ball_emitter"
        or receipt.get("status") != "pass"
    ):
        raise ValueError("accepted receipt identity or pass status is invalid")

    scene = receipt.get("scene_contract")
    if not isinstance(scene, Mapping) or scene.get("source_layers_unchanged") is not True:
        raise ValueError("accepted receipt source layer contract is invalid")
    records = scene.get("source_layer_records")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("accepted receipt must bind exactly two source layers")
    local_source_paths = {
        "fill_station.usda": source_stage,
        "geometry/fill_station.usdc": source_stage.parent
        / "geometry"
        / "fill_station.usdc",
    }
    for key, path in local_source_paths.items():
        if (
            not path.is_file()
            or path.stat().st_size != SOURCE_LAYER_BYTES[key]
            or _sha256(path) != SOURCE_LAYER_HASHES[key]
        ):
            raise ValueError("local source layer differs from the reviewed source bytes")
    source_hashes: dict[str, str] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "identifier",
            "resolved_path",
            "bytes",
            "sha256",
        }:
            raise ValueError("accepted receipt source layer record is malformed")
        identifier = raw_record.get("identifier")
        resolved_path = raw_record.get("resolved_path")
        if (
            not isinstance(identifier, str)
            or not identifier
            or not isinstance(resolved_path, str)
            or not resolved_path
        ):
            raise ValueError("accepted receipt source layer path is malformed")
        key = _source_layer_key(resolved_path)
        if key is None or _source_layer_key(identifier) != key:
            raise ValueError("accepted receipt source layer membership is invalid")
        if key in source_hashes:
            raise ValueError("accepted receipt source layer membership is duplicated")
        if (
            type(raw_record.get("bytes")) is not int
            or raw_record.get("bytes") <= 0
            or raw_record.get("sha256") != SOURCE_LAYER_HASHES[key]
            or raw_record.get("bytes") != SOURCE_LAYER_BYTES[key]
        ):
            raise ValueError("accepted receipt source layer hash binding is invalid")
        source_hashes[key] = SOURCE_LAYER_HASHES[key]
    if source_hashes != SOURCE_LAYER_HASHES:
        raise ValueError("accepted receipt source layer membership is invalid")

    render = receipt.get("render")
    if not isinstance(render, Mapping):
        raise ValueError("accepted receipt video manifest is missing")
    video_path = _artifact_path(case_root, render.get("video_path"), "accepted video")
    if not _valid_sha256(render.get("video_sha256")) or _sha256(video_path) != render.get(
        "video_sha256"
    ):
        raise ValueError("accepted video hash does not match its manifest")
    decoded_count = render.get("video_decoded_frame_count")
    if (
        render.get("video_decodable") is not True
        or isinstance(decoded_count, bool)
        or not isinstance(decoded_count, int)
        or decoded_count <= 0
    ):
        raise ValueError("accepted video manifest is not a successful decode")

    timing = receipt.get("timing")
    captures = receipt.get("captures")
    if not isinstance(timing, Mapping) or not isinstance(captures, list):
        raise ValueError("accepted receipt final capture manifest is missing")
    final_step = timing.get("primary_final_capture_step")
    if isinstance(final_step, bool) or not isinstance(final_step, int) or final_step <= 0:
        raise ValueError("accepted receipt final physics step is invalid")
    final_captures = [
        row
        for row in captures
        if (
            isinstance(row, Mapping)
            and type(row.get("physics_step")) is int
            and row.get("physics_step") == final_step
        )
    ]
    if len(final_captures) != 1:
        raise ValueError("accepted receipt must bind one final capture")
    snapshot_receipt = final_captures[0].get("particle_snapshot")
    if not isinstance(snapshot_receipt, Mapping):
        raise ValueError("accepted receipt final particle snapshot is missing")
    snapshot_path = _artifact_path(
        case_root, snapshot_receipt.get("snapshot_path"), "final snapshot"
    )
    if not _valid_sha256(snapshot_receipt.get("snapshot_sha256")) or _sha256(
        snapshot_path
    ) != snapshot_receipt.get("snapshot_sha256"):
        raise ValueError("final snapshot hash does not match its manifest")
    snapshot_value = _read_json(snapshot_path, "final snapshot")
    if not isinstance(snapshot_value, Mapping):
        raise ValueError("final snapshot must be a JSON object")
    if (
        snapshot_value.get("event_id") != snapshot_receipt.get("event_id")
        or snapshot_value.get("arrays_saved_before_assertions") is not True
        or snapshot_value.get("arrays") != snapshot_receipt.get("arrays")
    ):
        raise ValueError("final snapshot manifest differs from the accepted receipt")
    array_records = snapshot_value.get("arrays")
    if not isinstance(array_records, Mapping) or set(array_records) != _SNAPSHOT_ARRAY_NAMES:
        raise ValueError("final snapshot array membership is invalid")

    arrays: dict[str, np.ndarray] = {}
    array_paths: set[Path] = set()
    for name in sorted(_SNAPSHOT_ARRAY_NAMES):
        record = array_records[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"snapshot array {name} manifest is malformed")
        path = _artifact_path(case_root, record.get("path"), f"snapshot array {name}")
        if path in array_paths:
            raise ValueError("snapshot array paths must be unique")
        array_paths.add(path)
        if not _valid_sha256(record.get("sha256")) or _sha256(path) != record.get(
            "sha256"
        ):
            raise ValueError(f"snapshot array hash mismatch for {name}")
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"snapshot array {name} is unreadable") from exc
        shape = record.get("shape")
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or tuple(shape) != value.shape
            or record.get("dtype") != str(value.dtype)
            or record.get("finite") is not True
            or (value.dtype.kind in "fc" and not np.all(np.isfinite(value)))
        ):
            raise ValueError(f"snapshot array {name} manifest does not match its array")
        arrays[name] = value
    row_count = len(arrays["ids"])
    if row_count <= 0 or any(len(value) != row_count for value in arrays.values()):
        raise ValueError("snapshot arrays lost stable row alignment")
    vector_shapes = {
        "simulation_points": (row_count, 3),
        "positions": (row_count, 3),
        "velocities": (row_count, 3),
        "orientations": (row_count, 4),
        "scales": (row_count, 3),
        "birth_compositions": (row_count, 3),
        "current_compositions": (row_count, 3),
    }
    scalar_names = {
        "ids",
        "proto_indices",
        "scheduled_sources",
        "birth_steps",
        "birth_event_ids",
        "pour_event_ids",
    }
    if any(arrays[name].shape != shape for name, shape in vector_shapes.items()) or any(
        arrays[name].shape != (row_count,) for name in scalar_names
    ):
        raise ValueError("snapshot array has an invalid exact shape")
    float_names = set(vector_shapes)
    integer_names = {"ids", "proto_indices", "birth_steps"}
    string_names = {"scheduled_sources", "birth_event_ids", "pour_event_ids"}
    if (
        any(arrays[name].dtype != np.dtype(np.float64) for name in float_names)
        or any(arrays[name].dtype != np.dtype(np.int64) for name in integer_names)
        or any(arrays[name].dtype.kind != "U" for name in string_names)
    ):
        raise ValueError("snapshot array has an invalid exact dtype kind")
    if arrays["ids"].tolist() != list(range(row_count)):
        raise ValueError("snapshot arrays do not retain stable zero-based particle IDs")
    if np.any(arrays["proto_indices"] < 0) or np.any(arrays["birth_steps"] < 0):
        raise ValueError("snapshot array integer indices and steps must be non-negative")
    if np.any(arrays["scales"] <= 0.0):
        raise ValueError("snapshot array scales must be finite and strictly positive")
    for name in ("birth_compositions", "current_compositions"):
        compositions = arrays[name]
        if np.any(compositions < 0.0) or not np.allclose(
            np.sum(compositions, axis=1), 1.0, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(f"snapshot {name} must be non-negative and normalized")
    scheduled_sources = arrays["scheduled_sources"].tolist()
    birth_event_ids = arrays["birth_event_ids"].tolist()
    pour_event_ids = arrays["pour_event_ids"].tolist()
    if any(value not in {"t100", "t200", "t400"} for value in scheduled_sources):
        raise ValueError("snapshot array scheduled_sources has an invalid tank role")
    if any(
        not value.startswith("pour-")
        or "-event-" not in value
        or value.strip() != value
        for value in birth_event_ids
    ):
        raise ValueError("snapshot array birth_event_ids has an invalid event role")
    if any(
        not value.startswith("pour-")
        or len(value.split("-")) != 3
        or value.strip() != value
        for value in pour_event_ids
    ):
        raise ValueError("snapshot array pour_event_ids has an invalid event role")
    for value in arrays.values():
        value.setflags(write=False)

    return {
        "case_output": case_root,
        "stage_path": source_stage,
        "stage_sha256": _sha256(source_stage),
        "source_layer_hashes": source_hashes,
        "source_layer_paths": tuple(local_source_paths.values()),
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": _sha256(receipt_path),
        "video_path": video_path,
        "video_sha256": str(render["video_sha256"]),
        "final_physics_step": final_step,
        "snapshot_path": snapshot_path,
        "snapshot_sha256": str(snapshot_receipt["snapshot_sha256"]),
        "arrays": arrays,
        "part": "A",
        "excluded_from_model_inputs": True,
    }


def build_rigid_transport_plan(
    groups: Mapping[str, Collection[str]],
    initial_world_y_m: Mapping[str, float],
    spec: RigidTransportSpec = RigidTransportSpec(),
) -> tuple[dict[str, object], ...]:
    """Return exact smoothstep world-Y deltas without advancing source time."""

    if not isinstance(spec, RigidTransportSpec):
        raise TypeError("spec must be a RigidTransportSpec")
    if not isinstance(groups, Mapping) or {
        name: tuple(paths) for name, paths in groups.items()
    } != TRANSPORT_GROUPS:
        raise ValueError("transport groups differ from the reviewed exact groups")
    if not isinstance(initial_world_y_m, Mapping) or set(initial_world_y_m) != {
        ACTIVE_VIAL_PATH,
        INCOMING_VIAL_PATH,
        QUEUED_VIAL_PATH,
    }:
        raise ValueError("initial world-Y anchors must contain exactly three reviewed vials")
    anchor_y: dict[str, float] = {}
    for path, raw_value in initial_world_y_m.items():
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            raise ValueError("initial world-Y anchors must be finite numbers")
        anchor_y[path] = float(raw_value)
    active_y = anchor_y[ACTIVE_VIAL_PATH]
    incoming_y = anchor_y[INCOMING_VIAL_PATH]
    queued_y = anchor_y[QUEUED_VIAL_PATH]
    if not active_y > incoming_y > queued_y:
        raise ValueError("initial world-Y anchors are crossed")
    if not math.isclose(
        active_y - incoming_y,
        spec.incoming_final_delta_y_m,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("incoming vial does not end at the active world-Y anchor")
    frames: list[dict[str, object]] = []
    for frame_index in range(spec.frame_count):
        fraction = frame_index / (spec.frame_count - 1)
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        group_delta = {
            "outgoing": float(spec.outgoing_final_delta_y_m) * eased,
            "incoming": float(spec.incoming_final_delta_y_m) * eased,
            "queued": 0.0,
        }
        path_delta = {
            path: group_delta[group_name]
            for group_name in ("outgoing", "incoming", "queued")
            for path in groups[group_name]
        }
        group_delta_xyz = {
            group_name: [0.0, delta_y, 0.0]
            for group_name, delta_y in group_delta.items()
        }
        path_delta_xyz = {
            path: [0.0, delta_y, 0.0] for path, delta_y in path_delta.items()
        }
        frames.append(
            {
                "frame_index": frame_index,
                "source_time_code": 1,
                "time_s": frame_index / float(spec.fps),
                "smoothstep": eased,
                "group_delta_y_m": group_delta,
                "path_delta_y_m": path_delta,
                "group_delta_xyz_m": group_delta_xyz,
                "path_delta_xyz_m": path_delta_xyz,
            }
        )
    return tuple(frames)


def _validate_transport_plan(
    plan: Collection[Mapping[str, object]],
    groups: Mapping[str, Collection[str]],
    spec: RigidTransportSpec = RigidTransportSpec(),
) -> None:
    """Recompute every frozen plan scalar and transform independently."""

    if not isinstance(groups, Mapping) or {
        name: tuple(paths) for name, paths in groups.items()
    } != TRANSPORT_GROUPS:
        raise ValueError("strict transport plan groups differ from the reviewed groups")
    if not isinstance(plan, (list, tuple)) or len(plan) != spec.frame_count:
        raise ValueError("strict transport plan must contain exactly 21 rows")
    expected_paths = frozenset().union(*TRANSPORT_GROUPS.values())
    expected_row_keys = {
        "frame_index",
        "source_time_code",
        "time_s",
        "smoothstep",
        "group_delta_y_m",
        "path_delta_y_m",
        "group_delta_xyz_m",
        "path_delta_xyz_m",
    }
    for frame_index, row in enumerate(plan):
        fraction = frame_index / (spec.frame_count - 1)
        eased = fraction * fraction * (3.0 - 2.0 * fraction)
        expected_group_y = {
            "outgoing": float(spec.outgoing_final_delta_y_m) * eased,
            "incoming": float(spec.incoming_final_delta_y_m) * eased,
            "queued": 0.0,
        }
        expected_path_y = {
            path: expected_group_y[group_name]
            for group_name in ("outgoing", "incoming", "queued")
            for path in TRANSPORT_GROUPS[group_name]
        }
        expected_group_xyz = {
            name: [0.0, delta_y, 0.0]
            for name, delta_y in expected_group_y.items()
        }
        expected_path_xyz = {
            path: [0.0, delta_y, 0.0]
            for path, delta_y in expected_path_y.items()
        }
        if not isinstance(row, Mapping) or set(row) != expected_row_keys:
            raise ValueError("strict transport plan row schema is malformed")
        if type(row.get("frame_index")) is not int or row.get("frame_index") != frame_index:
            raise ValueError("strict transport plan frame indices are invalid")
        if type(row.get("source_time_code")) is not int or row.get(
            "source_time_code"
        ) != 1:
            raise ValueError("strict transport plan must freeze source time code 1")
        if type(row.get("time_s")) is not float or row.get("time_s") != (
            frame_index / float(spec.fps)
        ):
            raise ValueError("strict transport plan frame time is invalid")
        if type(row.get("smoothstep")) is not float or row.get("smoothstep") != eased:
            raise ValueError("strict transport plan smoothstep is invalid")
        for key, expected, expected_keys in (
            ("group_delta_y_m", expected_group_y, set(expected_group_y)),
            ("path_delta_y_m", expected_path_y, expected_paths),
        ):
            value = row.get(key)
            if (
                not isinstance(value, Mapping)
                or set(value) != expected_keys
                or any(type(item) is not float for item in value.values())
                or value != expected
            ):
                raise ValueError(f"strict transport plan {key} is invalid")
        for key, expected, expected_keys in (
            ("group_delta_xyz_m", expected_group_xyz, set(expected_group_xyz)),
            ("path_delta_xyz_m", expected_path_xyz, expected_paths),
        ):
            value = row.get(key)
            if (
                not isinstance(value, Mapping)
                or set(value) != expected_keys
                or any(
                    not isinstance(item, list)
                    or len(item) != 3
                    or any(type(component) is not float for component in item)
                    for item in value.values()
                )
                or value != expected
            ):
                raise ValueError(f"strict transport plan {key} is invalid")


def validate_authored_transport_diff(
    authored_diff: Mapping[str, object],
    groups: Mapping[str, Collection[str]],
    plan: Collection[Mapping[str, object]],
) -> None:
    """Reject every root/source edit and every unreviewed session property."""

    if not isinstance(authored_diff, Mapping):
        raise TypeError("authored_diff must be a mapping")
    if set(authored_diff) != {
        "edit_target",
        "source_time_code",
        "source_layer_property_paths",
        "root_layer_property_paths",
        "session_prim_paths",
        "count_body_reconstruction_prim_paths",
        "session_property_paths",
        "count_body_reconstruction_property_paths",
        "frame_index",
        "transport_xyz_by_path",
        "existing_xform_op_order_by_path",
        "authored_xform_op_order_by_path",
    }:
        raise ValueError("authored transport diff top-level schema is malformed")
    if not isinstance(groups, Mapping) or {
        name: tuple(paths) for name, paths in groups.items()
    } != TRANSPORT_GROUPS:
        raise ValueError("transport groups differ from the reviewed exact groups")
    _validate_transport_plan(plan, groups)
    if authored_diff.get("edit_target") != "anonymous_session_layer":
        raise ValueError("transport must target an anonymous session layer")
    source_time_code = authored_diff.get("source_time_code")
    if type(source_time_code) is not int or source_time_code != 1:
        raise ValueError("authored transport diff must freeze source time code 1")
    if authored_diff.get("source_layer_property_paths") != []:
        raise ValueError("source layer properties must remain unchanged")
    if authored_diff.get("root_layer_property_paths") != []:
        raise ValueError("root layer properties must remain unchanged")
    if authored_diff.get("session_prim_paths") != list(SESSION_PRIM_PATHS):
        raise ValueError("unauthorized session prim inventory was authored")
    if authored_diff.get("count_body_reconstruction_prim_paths") != list(
        COUNT_BODY_RECONSTRUCTION_PRIM_PATHS
    ):
        raise ValueError("count-body reconstruction prim inventory is invalid")
    frame_index = authored_diff.get("frame_index")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or frame_index < 0
        or frame_index >= len(plan)
    ):
        raise ValueError("authored transport frame index is invalid")
    expected_deltas = plan[frame_index].get("path_delta_y_m")
    transport_xyz = authored_diff.get("transport_xyz_by_path")
    if not isinstance(transport_xyz, Mapping) or set(transport_xyz) != _EXPECTED_PRIM_PATHS:
        raise ValueError("authored transport XYZ paths differ from the motion plan")
    for path, expected_y in expected_deltas.items():
        xyz = transport_xyz[path]
        if (
            not isinstance(xyz, list)
            or len(xyz) != 3
            or any(type(value) is not float or not math.isfinite(value) for value in xyz)
            or xyz[0] != 0.0
            or xyz[1] != expected_y
            or xyz[2] != 0.0
        ):
            raise ValueError("authored transport XYZ differs from the motion plan")

    existing_orders = authored_diff.get("existing_xform_op_order_by_path")
    authored_orders = authored_diff.get("authored_xform_op_order_by_path")
    if (
        not isinstance(existing_orders, Mapping)
        or not isinstance(authored_orders, Mapping)
        or set(existing_orders) != _EXPECTED_PRIM_PATHS
        or set(authored_orders) != _EXPECTED_PRIM_PATHS
    ):
        raise ValueError("transport xform op-order path inventory is invalid")
    for path in _EXPECTED_PRIM_PATHS:
        expected_existing = (
            ["xformOp:transform"]
            if path == COUNT_BODY_PATH
            else list(SOURCE_XFORM_OP_ORDER)
        )
        if existing_orders[path] != expected_existing or authored_orders[path] != [
            *expected_existing,
            TRANSPORT_XFORM_OP,
        ]:
            raise ValueError("transport xform op order was replaced or reordered")

    required_reconstruction = COUNT_BODY_RECONSTRUCTION_PROPERTY_PATHS
    reconstruction = authored_diff.get("count_body_reconstruction_property_paths")
    if (
        not isinstance(reconstruction, list)
        or len(reconstruction) != len(set(reconstruction))
        or reconstruction != sorted(required_reconstruction)
    ):
        raise ValueError("count-body reconstruction properties differ from the exact contract")
    session_properties = authored_diff.get("session_property_paths")
    if (
        not isinstance(session_properties, list)
        or len(session_properties) != len(set(session_properties))
        or session_properties
        != sorted(TRANSPORT_PROPERTY_PATHS | required_reconstruction)
    ):
        raise ValueError("unauthorized session properties were authored")


def validate_rigid_transport_receipt(
    receipt: Mapping[str, object],
    expected_inputs: Mapping[str, object],
    groups: Mapping[str, Collection[str]],
    plan: Collection[Mapping[str, object]],
) -> None:
    """Validate one complete Part A transport receipt."""

    if not all(isinstance(value, Mapping) for value in (receipt, expected_inputs, groups)):
        raise TypeError("receipt, expected_inputs and groups must be mappings")
    if not isinstance(plan, Collection):
        raise TypeError("plan must be a collection")
    _validate_transport_plan(plan, groups)
    if set(receipt) != {
        "schema_version",
        "artifact_type",
        "status",
        "part",
        "excluded_from_model_inputs",
        "spec",
        "bindings",
        "transport_groups",
        "frames",
        "authored_diff_sha256",
        "counters",
        "automatic_acceptance_gates",
    }:
        raise ValueError("transport receipt top-level schema is malformed")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != "rigid_vial_transport_receipt"
        or receipt.get("status") != "pass"
    ):
        raise ValueError("transport receipt identity is malformed")
    if (
        receipt.get("part") != "A"
        or receipt.get("excluded_from_model_inputs") is not True
    ):
        raise ValueError(
            "transport receipt must remain Part A provenance excluded from model inputs"
        )

    required_input_keys = {
        "stage_sha256",
        "source_layer_hashes",
        "receipt_sha256",
        "video_sha256",
        "snapshot_sha256",
        "final_physics_step",
        "authored_diff_sha256",
    }
    if set(expected_inputs) != required_input_keys:
        raise ValueError("expected transport input provenance binding is malformed")
    if (
        expected_inputs.get("stage_sha256") != SOURCE_LAYER_HASHES["fill_station.usda"]
        or expected_inputs.get("source_layer_hashes") != SOURCE_LAYER_HASHES
        or any(
            not _valid_sha256(expected_inputs.get(key))
            for key in (
                "receipt_sha256",
                "video_sha256",
                "snapshot_sha256",
                "authored_diff_sha256",
            )
        )
        or type(expected_inputs.get("final_physics_step")) is not int
        or expected_inputs.get("final_physics_step") <= 0
    ):
        raise ValueError("expected transport input provenance binding is invalid")
    expected_bindings = {
        "stage_sha256": expected_inputs["stage_sha256"],
        "source_layer_hashes": dict(expected_inputs["source_layer_hashes"]),
        "accepted_receipt_sha256": expected_inputs["receipt_sha256"],
        "accepted_video_sha256": expected_inputs["video_sha256"],
        "final_snapshot_sha256": expected_inputs["snapshot_sha256"],
        "final_physics_step": expected_inputs["final_physics_step"],
    }
    bindings = receipt.get("bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != set(expected_bindings)
        or type(bindings.get("final_physics_step")) is not int
        or bindings.get("final_physics_step") <= 0
        or bindings != expected_bindings
    ):
        raise ValueError("transport receipt provenance binding is wrong")

    expected_groups = {name: list(paths) for name, paths in TRANSPORT_GROUPS.items()}
    if (
        {name: tuple(paths) for name, paths in groups.items()} != TRANSPORT_GROUPS
        or receipt.get("transport_groups") != expected_groups
    ):
        raise ValueError("transport receipt group binding is wrong")
    if not isinstance(plan, (list, tuple)) or len(plan) != 21:
        raise ValueError("transport receipt plan binding is malformed")
    expected_spec = {
        "frame_count": 21,
        "fps": 10.0,
        "outgoing_final_delta_y_m": plan[-1]["group_delta_y_m"]["outgoing"],
        "incoming_final_delta_y_m": INCOMING_FINAL_DELTA_Y_M,
    }
    receipt_spec = receipt.get("spec")
    if (
        not isinstance(receipt_spec, Mapping)
        or set(receipt_spec) != set(expected_spec)
        or type(receipt_spec.get("frame_count")) is not int
        or type(receipt_spec.get("fps")) is not float
        or type(receipt_spec.get("outgoing_final_delta_y_m")) is not float
        or type(receipt_spec.get("incoming_final_delta_y_m")) is not float
        or any(
            not math.isfinite(receipt_spec[name])
            for name in (
                "fps",
                "outgoing_final_delta_y_m",
                "incoming_final_delta_y_m",
            )
        )
        or receipt_spec != expected_spec
    ):
        raise ValueError("transport receipt spec binding is wrong")

    frames = receipt.get("frames")
    if not isinstance(frames, list) or len(frames) != 21:
        raise ValueError("transport receipt frame manifest is malformed")
    frame_hashes: list[str] = []
    for index, (frame, planned) in enumerate(zip(frames, plan)):
        if not isinstance(frame, Mapping):
            raise ValueError("transport receipt frame manifest is malformed")
        digest = frame.get("rgb_sha256")
        if (
            set(frame)
            != {
                "frame_index",
                "source_time_code",
                "time_s",
                "smoothstep",
                "rgb_sha256",
                "group_delta_y_m",
                "path_delta_y_m",
                "group_delta_xyz_m",
                "path_delta_xyz_m",
            }
            or type(frame.get("frame_index")) is not int
            or frame.get("frame_index") != index
            or type(frame.get("source_time_code")) is not int
            or frame.get("source_time_code") != 1
            or type(frame.get("time_s")) is not float
            or frame.get("time_s") != planned.get("time_s")
            or type(frame.get("smoothstep")) is not float
            or frame.get("smoothstep") != planned.get("smoothstep")
            or frame.get("group_delta_y_m") != planned.get("group_delta_y_m")
            or frame.get("path_delta_y_m") != planned.get("path_delta_y_m")
            or frame.get("group_delta_xyz_m") != planned.get("group_delta_xyz_m")
            or frame.get("path_delta_xyz_m") != planned.get("path_delta_xyz_m")
            or not _valid_sha256(digest)
        ):
            raise ValueError("transport receipt frame manifest is malformed")
        group_delta = frame["group_delta_y_m"]
        path_delta = frame["path_delta_y_m"]
        group_xyz = frame["group_delta_xyz_m"]
        path_xyz = frame["path_delta_xyz_m"]
        if (
            any(type(value) is not float for value in group_delta.values())
            or any(type(value) is not float for value in path_delta.values())
            or any(
                not isinstance(xyz, list)
                or len(xyz) != 3
                or any(type(component) is not float for component in xyz)
                for xyz in (*group_xyz.values(), *path_xyz.values())
            )
            or any(
                not math.isfinite(value)
                for value in (
                    *group_delta.values(),
                    *path_delta.values(),
                    *(component for xyz in group_xyz.values() for component in xyz),
                    *(component for xyz in path_xyz.values() for component in xyz),
                )
            )
        ):
            raise ValueError("transport receipt frame manifest contains non-finite values")
        frame_hashes.append(str(digest))
    if len(set(frame_hashes)) != 21:
        raise ValueError("transport receipt frame manifest hashes are not unique")
    if receipt.get("authored_diff_sha256") != expected_inputs["authored_diff_sha256"]:
        raise ValueError("transport receipt authored diff binding is wrong")

    counters = receipt.get("counters")
    integer_counter_names = (
        "physics_steps",
        "direct_step_calls",
        "physics_callbacks",
    )
    expected_counter_keys = {
        f"{name}_{suffix}"
        for name in integer_counter_names
        for suffix in ("before", "after", "during_transport")
    } | {
        "simulation_time_s_before",
        "simulation_time_s_after",
        "timeline_time_s_before",
        "timeline_time_s_after",
        "timeline_playing_before",
        "timeline_playing_after",
        "source_time_code_before",
        "source_time_code_after",
    }
    if not isinstance(counters, Mapping) or set(counters) != expected_counter_keys:
        raise ValueError("transport receipt counter contract is missing")
    for name in integer_counter_names:
        before = counters.get(f"{name}_before")
        after = counters.get(f"{name}_after")
        during = counters.get(f"{name}_during_transport")
        if (
            type(before) is not int
            or type(after) is not int
            or type(during) is not int
            or before < 0
            or after < 0
            or before != after
            or during != 0
        ):
            raise ValueError("transport receipt counter contract advanced physics")
    if counters.get("physics_steps_before") != expected_inputs["final_physics_step"]:
        raise ValueError("transport receipt counter contract changed the accepted step")
    for clock in ("simulation_time_s", "timeline_time_s"):
        before = counters.get(f"{clock}_before")
        after = counters.get(f"{clock}_after")
        if (
            type(before) is not float
            or type(after) is not float
            or not math.isfinite(before)
            or not math.isfinite(after)
            or before < 0.0
            or after < 0.0
            or before != after
        ):
            raise ValueError("transport receipt counter contract advanced a clock")
    if (
        counters.get("timeline_playing_before") is not False
        or counters.get("timeline_playing_after") is not False
    ):
        raise ValueError("transport receipt counter contract played the source timeline")
    if (
        type(counters.get("source_time_code_before")) is not int
        or type(counters.get("source_time_code_after")) is not int
        or counters.get("source_time_code_before") != 1
        or counters.get("source_time_code_after") != 1
    ):
        raise ValueError("transport receipt counter contract changed source time code 1")

    required_gates = {
        "source_and_receipt_hashes_unchanged",
        "snapshot_binding",
        "session_diff_allowlisted",
        "zero_clock_and_physics_advancement",
        "outgoing_absent_final",
        "incoming_centered_final",
        "fixed_background_sentinel",
        "start_splice",
        "end_splice",
        "unique_frame_hashes",
    }
    gates = receipt.get("automatic_acceptance_gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != required_gates
        or any(value is not True for value in gates.values())
    ):
        raise ValueError("transport receipt automatic acceptance gates are incomplete")

"""Fail-closed contracts for an exact PhysX-isosurface mesh proxy.

This module is deliberately independent of Isaac Sim.  A Kit-side adapter must
read and author the actual ``UsdGeom.Mesh`` objects; these helpers define the
canonical geometry evidence and the no-hide-before-parity transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral, Real
import re
import struct
from collections.abc import Mapping
from typing import Protocol


@dataclass(frozen=True)
class MeshSnapshot:
    """Canonical float32 geometry, int32 topology, and float64 transform."""

    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    local_transform: tuple[float, ...]
    points_sha256: str
    face_vertex_counts_sha256: str
    face_vertex_indices_sha256: str
    topology_sha256: str
    local_transform_sha256: str
    mesh_sha256: str

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def face_count(self) -> int:
        return len(self.face_vertex_counts)

    @property
    def corner_count(self) -> int:
        return len(self.face_vertex_indices)


@dataclass(frozen=True)
class LiquidProxyIntent:
    source_path: str
    proxy_path: str
    material_path: str
    semantic_type: str = "class"
    semantic_labels: tuple[str, ...] = ("liquid",)


class MeshProxyAdapter(Protocol):
    """Minimum Kit-side interface needed by :func:`copy_exact_mesh_proxy`."""

    def read_mesh(self, path: str) -> object: ...

    def prim_exists(self, path: str) -> bool: ...

    def define_standard_mesh(self, path: str) -> None: ...

    def write_mesh(self, path: str, snapshot: MeshSnapshot) -> None: ...

    def bind_material(self, path: str, material_path: str) -> None: ...

    def bind_semantics(
        self, path: str, semantic_type: str, labels: tuple[str, ...]
    ) -> None: ...

    def read_material_binding(self, path: str) -> object: ...

    def read_semantics(self, path: str, semantic_type: str) -> object: ...


PROXY_RGB_PARITY_THRESHOLDS = {
    "mean_abs_rgb_max": 2.0,
    "p95_abs_rgb_max": 8.0,
    "fraction_abs_rgb_gt_16_max": 0.01,
}


def build_liquid_proxy_intent(
    *, source_path: str, proxy_path: str, material_path: str
) -> LiquidProxyIntent:
    paths = {
        "source_path": source_path,
        "proxy_path": proxy_path,
        "material_path": material_path,
    }
    component = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for name, path in paths.items():
        if not isinstance(path, str):
            raise TypeError(f"{name} must be a string")
        if (
            not path.startswith("/")
            or path == "/"
            or path.endswith("/")
            or any(not component.fullmatch(part) for part in path[1:].split("/"))
        ):
            raise ValueError(f"{name} must be an absolute USD prim path")
    if source_path == proxy_path:
        raise ValueError("source and proxy paths must differ")
    if source_path.rpartition("/")[0] != proxy_path.rpartition("/")[0]:
        raise ValueError("source and proxy must have the same parent transform")
    return LiquidProxyIntent(
        source_path=source_path,
        proxy_path=proxy_path,
        material_path=material_path,
    )


def _snapshot_from_record(record: object, *, role: str) -> MeshSnapshot:
    if not isinstance(record, Mapping):
        raise TypeError(f"{role} mesh readback must be a mapping")
    if record.get("schema_type") != "Mesh":
        raise ValueError(f"{role} prim must read back as a standard Mesh")
    required = (
        "points",
        "face_vertex_counts",
        "face_vertex_indices",
        "local_transform",
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"{role} mesh readback missing {', '.join(missing)}")
    return canonical_mesh_snapshot(
        record["points"],
        record["face_vertex_counts"],
        record["face_vertex_indices"],
        record["local_transform"],
    )


def copy_exact_mesh_proxy(
    adapter: MeshProxyAdapter, intent: LiquidProxyIntent
) -> dict[str, object]:
    if not isinstance(intent, LiquidProxyIntent):
        raise TypeError("intent must be a LiquidProxyIntent")
    source = _snapshot_from_record(
        adapter.read_mesh(intent.source_path), role="source"
    )
    if adapter.prim_exists(intent.proxy_path):
        raise ValueError("proxy path already exists; no replacement is allowed")

    adapter.define_standard_mesh(intent.proxy_path)
    adapter.write_mesh(intent.proxy_path, source)
    adapter.bind_material(intent.proxy_path, intent.material_path)
    adapter.bind_semantics(
        intent.proxy_path, intent.semantic_type, intent.semantic_labels
    )

    proxy = _snapshot_from_record(adapter.read_mesh(intent.proxy_path), role="proxy")
    if proxy != source:
        raise ValueError("proxy geometry or transform readback differs from source")
    material = adapter.read_material_binding(intent.proxy_path)
    if material != intent.material_path:
        raise ValueError("proxy material binding readback differs from intent")
    labels = adapter.read_semantics(intent.proxy_path, intent.semantic_type)
    try:
        label_tuple = tuple(labels)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("proxy semantic binding readback is missing") from exc
    if label_tuple != intent.semantic_labels:
        raise ValueError("proxy semantic binding readback differs from intent")

    return {
        "status": "proxy_authored_pending_rgb_parity",
        "source_path": intent.source_path,
        "proxy_path": intent.proxy_path,
        "same_parent_transform": True,
        "point_count": source.point_count,
        "face_count": source.face_count,
        "corner_count": source.corner_count,
        "points_sha256": source.points_sha256,
        "face_vertex_counts_sha256": source.face_vertex_counts_sha256,
        "face_vertex_indices_sha256": source.face_vertex_indices_sha256,
        "topology_sha256": source.topology_sha256,
        "local_transform_sha256": source.local_transform_sha256,
        "mesh_sha256": source.mesh_sha256,
        "proxy_points_sha256": proxy.points_sha256,
        "proxy_face_vertex_counts_sha256": proxy.face_vertex_counts_sha256,
        "proxy_face_vertex_indices_sha256": proxy.face_vertex_indices_sha256,
        "proxy_topology_sha256": proxy.topology_sha256,
        "proxy_local_transform_sha256": proxy.local_transform_sha256,
        "proxy_mesh_sha256": proxy.mesh_sha256,
        "exact_geometry_match": True,
        "material_binding": material,
        "semantic_binding": {
            "type": intent.semantic_type,
            "labels": list(intent.semantic_labels),
        },
        "source_visibility": "unchanged",
        "source_hidden": False,
        "shared_rgb_plan": {
            "parity_gate_status": "not_run",
            "hide_source_allowed": False,
            "required_next_gate": (
                "same-state original-isosurface versus proxy RGB parity with saved "
                "images, hashes, and frozen thresholds"
            ),
            "after_parity": (
                "hide the generated source, show and label the exact proxy, then acquire "
                "one shared RGB and semantic capture"
            ),
        },
        "claim_limits": {
            "adapter_contract_only": True,
            "rgb_parity_observed": False,
            "semantic_mask_observed": False,
            "training_eligible": False,
        },
    }


def evaluate_proxy_rgb_parity_for_source_hide(
    proxy_evidence: object, parity_evidence: object
) -> dict[str, object]:
    """Evaluate saved same-state RGB evidence without changing USD visibility."""

    if not isinstance(proxy_evidence, Mapping):
        raise TypeError("proxy_evidence must be a mapping")
    if not isinstance(parity_evidence, Mapping):
        raise TypeError("parity_evidence must be a mapping")
    if (
        proxy_evidence.get("status") != "proxy_authored_pending_rgb_parity"
        or proxy_evidence.get("exact_geometry_match") is not True
        or proxy_evidence.get("source_hidden") is not False
    ):
        raise ValueError("exact visible-source proxy evidence is required")
    for source_key, proxy_key in (
        ("points_sha256", "proxy_points_sha256"),
        ("topology_sha256", "proxy_topology_sha256"),
        ("local_transform_sha256", "proxy_local_transform_sha256"),
        ("mesh_sha256", "proxy_mesh_sha256"),
    ):
        source_hash = proxy_evidence.get(source_key)
        proxy_hash = proxy_evidence.get(proxy_key)
        if not (_valid_sha256(source_hash) and source_hash == proxy_hash):
            raise ValueError("proxy geometry hash evidence is missing or inconsistent")

    if parity_evidence.get("same_physics_step") is not True:
        raise ValueError("RGB parity must use the same physics step")
    if parity_evidence.get("same_renderer_settings") is not True:
        raise ValueError("RGB parity must use the same renderer settings")
    if parity_evidence.get("parity_capture_isolated") is not True:
        raise ValueError("RGB parity must run in an isolated disposable capture state")
    if parity_evidence.get("source_restored_visible") is not True:
        raise ValueError("source visibility must be restored before the parity decision")
    if parity_evidence.get("thresholds") != PROXY_RGB_PARITY_THRESHOLDS:
        raise ValueError("RGB parity thresholds differ from the frozen contract")
    for name in ("original_rgb_sha256", "proxy_rgb_sha256", "absdiff_sha256"):
        if not _valid_sha256(parity_evidence.get(name)):
            raise ValueError(f"{name} must be a lowercase SHA-256")

    metrics: dict[str, float] = {}
    for name in ("mean_abs_rgb", "p95_abs_rgb", "fraction_abs_rgb_gt_16"):
        raw = parity_evidence.get(name)
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(f"{name} must be a real number")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        metrics[name] = value
    if metrics["fraction_abs_rgb_gt_16"] > 1.0:
        raise ValueError("fraction_abs_rgb_gt_16 cannot exceed one")

    failures: list[str] = []
    if metrics["mean_abs_rgb"] > PROXY_RGB_PARITY_THRESHOLDS["mean_abs_rgb_max"]:
        failures.append("mean_abs_rgb_exceeds_threshold")
    if metrics["p95_abs_rgb"] > PROXY_RGB_PARITY_THRESHOLDS["p95_abs_rgb_max"]:
        failures.append("p95_abs_rgb_exceeds_threshold")
    if (
        metrics["fraction_abs_rgb_gt_16"]
        > PROXY_RGB_PARITY_THRESHOLDS["fraction_abs_rgb_gt_16_max"]
    ):
        failures.append("fraction_abs_rgb_gt_16_exceeds_threshold")
    passed = not failures
    return {
        "status": (
            "rgb_parity_pass_pending_shared_capture"
            if passed
            else "rgb_parity_failed_source_must_remain_visible"
        ),
        "hide_source_allowed": passed,
        "source_visibility_change_executed": False,
        "proxy_visibility": "visible",
        "next_capture": (
            "one_shared_rgb_semantic_capture" if passed else None
        ),
        "semantic_mask_observed": False,
        "thresholds": dict(PROXY_RGB_PARITY_THRESHOLDS),
        "metrics": metrics,
        "artifact_sha256": {
            name: parity_evidence[name]
            for name in (
                "original_rgb_sha256",
                "proxy_rgb_sha256",
                "absdiff_sha256",
            )
        },
        "failure_reasons": failures,
        "claim_limits": {
            "geometry_proxy_exact": True,
            "rgb_parity_evidence_observed": True,
            "rgb_parity_passed": passed,
            "semantic_mask_observed": False,
            "training_eligible": False,
        },
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _items(value: object, name: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray, dict)):
        raise TypeError(f"{name} must be a non-string sequence")
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence") from exc


def _float32(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} values must be real numbers")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} values must be finite")
    try:
        rounded = struct.unpack("<f", struct.pack("<f", converted))[0]
    except OverflowError as exc:
        raise ValueError(f"{name} values must fit finite float32") from exc
    if not math.isfinite(rounded):
        raise ValueError(f"{name} values must fit finite float32")
    return rounded


def _float64(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} values must be real numbers")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} values must be finite")
    return converted


def _int32(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} values must be integers")
    converted = int(value)
    if not -(2**31) <= converted < 2**31:
        raise ValueError(f"{name} values must fit signed int32")
    return converted


def _digest(tag: str, *payloads: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(tag.encode("ascii"))
    digest.update(b"\0")
    for payload in payloads:
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def canonical_mesh_snapshot(
    points: object,
    face_vertex_counts: object,
    face_vertex_indices: object,
    local_transform: object,
) -> MeshSnapshot:
    """Return a validated, deterministically hashed mesh snapshot."""

    point_rows = _items(points, "points")
    if not point_rows:
        raise ValueError("points must be nonempty")
    canonical_points: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(point_rows):
        coordinates = _items(row, f"points[{row_index}]")
        if len(coordinates) != 3:
            raise ValueError("every point must have exactly three coordinates")
        canonical_points.append(
            tuple(_float32(value, "points") for value in coordinates)  # type: ignore[arg-type]
        )

    canonical_counts = tuple(
        _int32(value, "face_vertex_counts")
        for value in _items(face_vertex_counts, "face_vertex_counts")
    )
    canonical_indices = tuple(
        _int32(value, "face_vertex_indices")
        for value in _items(face_vertex_indices, "face_vertex_indices")
    )
    if not canonical_counts:
        raise ValueError("face_vertex_counts must be nonempty")
    if any(count < 3 for count in canonical_counts):
        raise ValueError("every mesh face must have at least three vertices")
    if sum(canonical_counts) != len(canonical_indices):
        raise ValueError("face counts must sum to the number of face indices")
    if any(index < 0 or index >= len(canonical_points) for index in canonical_indices):
        raise ValueError("face indices must reference existing points")

    transform_rows = _items(local_transform, "local_transform")
    if len(transform_rows) == 4 and all(
        not isinstance(row, Real) for row in transform_rows
    ):
        flattened: list[object] = []
        for row in transform_rows:
            values = _items(row, "local_transform row")
            if len(values) != 4:
                raise ValueError("local_transform must be a 4x4 matrix")
            flattened.extend(values)
    else:
        flattened = transform_rows
    if len(flattened) != 16:
        raise ValueError("local_transform must contain 16 values")
    canonical_transform = tuple(
        _float64(value, "local_transform") for value in flattened
    )

    point_bytes = b"".join(struct.pack("<3f", *point) for point in canonical_points)
    count_bytes = b"".join(struct.pack("<i", value) for value in canonical_counts)
    index_bytes = b"".join(struct.pack("<i", value) for value in canonical_indices)
    transform_bytes = struct.pack("<16d", *canonical_transform)
    points_hash = _digest("fluidmix.proxy.points.v1", point_bytes)
    counts_hash = _digest("fluidmix.proxy.face-counts.v1", count_bytes)
    indices_hash = _digest("fluidmix.proxy.face-indices.v1", index_bytes)
    topology_hash = _digest(
        "fluidmix.proxy.topology.v1",
        bytes.fromhex(counts_hash),
        bytes.fromhex(indices_hash),
    )
    transform_hash = _digest("fluidmix.proxy.local-transform.v1", transform_bytes)
    mesh_hash = _digest(
        "fluidmix.proxy.mesh.v1",
        bytes.fromhex(points_hash),
        bytes.fromhex(topology_hash),
        bytes.fromhex(transform_hash),
    )
    return MeshSnapshot(
        points=tuple(canonical_points),
        face_vertex_counts=canonical_counts,
        face_vertex_indices=canonical_indices,
        local_transform=canonical_transform,
        points_sha256=points_hash,
        face_vertex_counts_sha256=counts_hash,
        face_vertex_indices_sha256=indices_hash,
        topology_sha256=topology_hash,
        local_transform_sha256=transform_hash,
        mesh_sha256=mesh_hash,
    )

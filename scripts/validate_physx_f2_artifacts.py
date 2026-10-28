#!/usr/bin/env python3
"""Validate copied-back schema-14 F2 artifacts against their receipt."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fluidmix.isosurface_mesh_proxy import PROXY_RGB_PARITY_THRESHOLDS
from fluidmix.paint_ball_emitter import (
    COUNT_DRIVEN_RENDER_PATH,
    EMITTER_RENDER_PATH,
    PERSISTENT_RENDER_STREAM_PATH,
    attach_visible_volume_evidence,
    validate_paint_ball_receipt,
)
from fluidmix.physx_volume_gate import (
    FILL_CAM_CARRIER_OCCLUSION_CAMERA_PATH,
    expected_carrier_occluder_state,
    load_fillcam_carrier_occlusion_profile,
)
from fluidmix.transform_attestation import (
    attest_effective_state_matrix,
    verify_effective_state_matrix_attestation,
)
from fluidmix.physx_three_pour import (
    FLUIDMIX_NAMESPACE,
    GENERATED_ISOSURFACE_PATH,
    LIQUID_PROXY_PATH,
    SOURCE_LIQUID_BOTTOM_M,
    SOURCE_LIQUID_RADIUS_M,
    VIAL_CENTER_XY_M,
    VIAL_MOUTH_M,
    validate_three_pour_receipt,
)
from fluidmix.render_capture import (
    LabelMappingError,
    exact_instance_semantic_overlap,
    exact_semantic_class_mask,
)
from physx_three_pour_sequence import (
    _realtime_visible_height_gate_valid,
    _validate_realtime_height_banded_receipt,
)


class ArtifactValidationError(ValueError):
    """One copied artifact is missing, unsafe, corrupt, or inconsistent."""


RUNTIME_PARTICLE_ARRAYS = {
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


def resolve_artifact(root: Path, relative_path: object) -> Path:
    """Resolve one existing regular receipt artifact without path traversal."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact path must be a non-empty string")
    untrusted = Path(relative_path)
    if untrusted.is_absolute() or relative_path == "." or ".." in untrusted.parts:
        raise ValueError(f"artifact path is not a safe relative file: {relative_path!r}")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"artifact root is missing or unreadable: {root}") from exc
    if not resolved_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    try:
        resolved = (resolved_root / untrusted).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"artifact is missing or unreadable: {relative_path!r}") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"artifact escapes root or is not a file: {relative_path!r}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: object) -> None:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ArtifactValidationError(f"invalid expected SHA-256 for {path.name}")
    observed = sha256_file(path)
    if observed != expected:
        raise ArtifactValidationError(
            f"SHA-256 mismatch for {path.name}: {observed} != {expected}"
        )


def read_rgb(path: Path, expected_width: int, expected_height: int):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ArtifactValidationError(f"RGB is unreadable: {path.name}")
    expected_shape = (expected_height, expected_width, 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise ArtifactValidationError(
            f"RGB shape/dtype mismatch for {path.name}: {image.shape}/{image.dtype}"
        )
    return image


def read_mask(path: Path, expected_width: int, expected_height: int):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ArtifactValidationError(f"mask is unreadable: {path.name}")
    expected_shape = (expected_height, expected_width)
    if mask.shape != expected_shape or mask.dtype != np.uint8:
        raise ArtifactValidationError(
            f"mask shape/dtype mismatch for {path.name}: {mask.shape}/{mask.dtype}"
        )
    if int(np.count_nonzero(mask)) <= 0:
        raise ArtifactValidationError(f"mask is not nonempty: {path.name}")
    if not set(int(value) for value in np.unique(mask)).issubset({0, 255}):
        raise ArtifactValidationError(f"mask is not binary 0/255: {path.name}")
    return mask


def read_semantic(path: Path, expected_width: int, expected_height: int):
    try:
        semantic = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ArtifactValidationError(f"semantic array is unreadable: {path.name}") from exc
    expected_shape = (expected_height, expected_width)
    if semantic.shape != expected_shape:
        raise ArtifactValidationError(
            f"semantic shape mismatch for {path.name}: {semantic.shape}"
        )
    if np.issubdtype(semantic.dtype, np.bool_) or not np.issubdtype(
        semantic.dtype, np.integer
    ):
        raise ArtifactValidationError(
            f"semantic dtype is not integer for {path.name}: {semantic.dtype}"
        )
    if np.any(semantic < 0):
        raise ArtifactValidationError(f"semantic IDs are negative: {path.name}")
    return semantic


def read_particle_array(path: Path) -> np.ndarray:
    """Load one lossless runtime array and reject object/non-finite payloads."""

    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ArtifactValidationError(
            f"particle array is unreadable: {path.name}"
        ) from exc
    if values.dtype.kind == "O":
        raise ArtifactValidationError(f"particle array has object dtype: {path.name}")
    if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
        raise ArtifactValidationError(f"particle array is non-finite: {path.name}")
    return values


def validate_semantic_mask_evidence(
    semantic: np.ndarray,
    mask: np.ndarray,
    pair: Mapping[str, Any],
) -> None:
    """Prove a saved liquid mask is exactly the renderer semantic-class union."""

    raw_mapping = pair.get("semantic_id_to_class_labels")
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise ArtifactValidationError("semantic receipt mapping is malformed")
    annotator_mapping: dict[str, dict[str, str]] = {}
    expected_mapping: list[tuple[int, tuple[str, ...]]] = []
    for raw_id, raw_labels in raw_mapping.items():
        if (
            not isinstance(raw_id, str)
            or re.fullmatch(r"[0-9]+", raw_id) is None
            or str(int(raw_id, 10)) != raw_id
            or not isinstance(raw_labels, list)
            or not raw_labels
            or not all(
                isinstance(label, str)
                and bool(label)
                and label.strip() == label
                and "," not in label
                for label in raw_labels
            )
            or len(set(raw_labels)) != len(raw_labels)
        ):
            raise ArtifactValidationError("semantic receipt mapping is malformed")
        annotator_mapping[raw_id] = {"class": ", ".join(raw_labels)}
        expected_mapping.append((int(raw_id, 10), tuple(raw_labels)))

    try:
        exact = exact_semantic_class_mask(
            semantic,
            annotator_mapping,
            class_name="liquid",
        )
    except LabelMappingError as exc:
        raise ArtifactValidationError(f"semantic receipt mapping/mask is invalid: {exc}") from exc

    expected_mapping_tuple = tuple(sorted(expected_mapping))
    if exact.id_to_labels != expected_mapping_tuple:
        raise ArtifactValidationError("semantic receipt mapping differs from artifact")
    if pair.get("semantic_unique_ids") != list(exact.visible_semantic_ids):
        raise ArtifactValidationError("semantic unique IDs differ from artifact")
    if pair.get("liquid_semantic_ids") != list(exact.semantic_ids):
        raise ArtifactValidationError("liquid semantic IDs differ from mapping")
    if not np.array_equal(mask, exact.mask):
        raise ArtifactValidationError("mask differs from exact semantic class IDs")

    expected_counts = {
        str(semantic_id): pixel_count
        for semantic_id, pixel_count in exact.pixel_counts_by_id
    }
    if pair.get("liquid_pixel_counts_by_id") != expected_counts:
        raise ArtifactValidationError(
            "per-ID pixel counts differ from semantic artifact"
        )
    if pair.get("liquid_pixel_count") != exact.pixel_count:
        raise ArtifactValidationError(
            "liquid pixel count differs from semantic artifact"
        )


def validate_video(
    path: Path,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ArtifactValidationError(f"video is unreadable: {path.name}")
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (expected_height, expected_width, 3):
                raise ArtifactValidationError(
                    f"video resolution mismatch for {path.name}: {frame.shape}"
                )
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_frames:
        raise ArtifactValidationError(
            f"video frame count mismatch for {path.name}: {decoded} != {expected_frames}"
        )
    return decoded


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _load_json(path: Path) -> object:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"JSON is unreadable: {path.name}") from exc


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _exact_proxy_geometry(value: object, *, status: str) -> bool:
    row = _mapping(value)
    return bool(
        row.get("status") == status
        and row.get("source_path") == GENERATED_ISOSURFACE_PATH
        and row.get("proxy_path") == LIQUID_PROXY_PATH
        and row.get("same_parent_transform") is True
        and row.get("exact_geometry_match") is True
        and isinstance(row.get("point_count"), int)
        and not isinstance(row.get("point_count"), bool)
        and int(row.get("point_count", 0)) > 0
        and isinstance(row.get("face_count"), int)
        and not isinstance(row.get("face_count"), bool)
        and int(row.get("face_count", 0)) > 0
        and all(
            _valid_sha256(row.get(source_key))
            and row.get(source_key) == row.get(proxy_key)
            for source_key, proxy_key in (
                ("points_sha256", "proxy_points_sha256"),
                ("face_vertex_counts_sha256", "proxy_face_vertex_counts_sha256"),
                ("face_vertex_indices_sha256", "proxy_face_vertex_indices_sha256"),
                ("topology_sha256", "proxy_topology_sha256"),
                ("local_transform_sha256", "proxy_local_transform_sha256"),
                ("mesh_sha256", "proxy_mesh_sha256"),
            )
        )
        and row.get("material_binding")
        == f"{FLUIDMIX_NAMESPACE}/GlobalLiquidMaterial"
        and row.get("semantic_binding")
        == {"type": "class", "labels": ["liquid"]}
    )


def validate_mesh_proxy_artifacts(
    app_output: Path,
    receipt: Mapping[str, Any],
    *,
    expected_width: int,
    expected_height: int,
) -> tuple[set[Path], list[str]]:
    """Verify schema-14 proxy bytes after an optional stream-only prefix."""

    failures: list[str] = []
    checked_paths: set[Path] = set()
    verified_hashes: dict[Path, str] = {}

    def fail(message: str) -> None:
        if message not in failures:
            failures.append(message)

    def checked(
        label: str,
        relative_path: object,
        expected_hash: object,
        reader: Callable[[Path], object],
    ) -> tuple[Path, object] | None:
        try:
            path = resolve_artifact(app_output, relative_path)
            if path in verified_hashes and verified_hashes[path] != expected_hash:
                raise ArtifactValidationError(
                    f"conflicting receipt hashes for {path.name}"
                )
            verify_hash(path, expected_hash)
            verified_hashes[path] = str(expected_hash)
            checked_paths.add(path)
            return path, reader(path)
        except (ValueError, ArtifactValidationError, OSError) as exc:
            fail(f"artifact:mesh_proxy:{label}:{exc}")
            return None

    mesh_proxy = _mapping(receipt.get("mesh_proxy"))
    stage_surface_sync_records = mesh_proxy.get("stage_surface_sync_records")
    if stage_surface_sync_records != []:
        fail("artifact:mesh_proxy:stage_surface_sync:receipt contract invalid")
    lifecycle_records = mesh_proxy.get(
        "stage_isosurface_lifecycle_refresh_records"
    )
    if not isinstance(lifecycle_records, list) or len(lifecycle_records) > 1:
        fail("artifact:mesh_proxy:stage_lifecycle_refresh:receipt contract invalid")
        lifecycle_records = []
    for index, raw_record in enumerate(lifecycle_records):
        record = _mapping(raw_record)
        refresh = checked(
            f"stage_lifecycle_refresh[{index}]",
            record.get("artifact_path"),
            record.get("artifact_sha256"),
            _load_json,
        )
        expected_refresh = {
            key: value
            for key, value in record.items()
            if key not in {"artifact_path", "artifact_sha256"}
        }
        if refresh is not None and refresh[1] != expected_refresh:
            fail(
                f"artifact:mesh_proxy:stage_lifecycle_refresh[{index}]:content mismatch"
            )
    creation_evidence = mesh_proxy.get("creation_evidence")
    if not _exact_proxy_geometry(
        creation_evidence, status="proxy_authored_pending_rgb_parity"
    ):
        fail("artifact:mesh_proxy:creation:exact geometry evidence invalid")
    creation = checked(
        "creation",
        mesh_proxy.get("creation_artifact_path"),
        mesh_proxy.get("creation_artifact_sha256"),
        _load_json,
    )
    expected_creation = {
        "physics_step": mesh_proxy.get("creation_step"),
        "evidence": creation_evidence,
        "source_visibility": "inherited",
        "proxy_visibility": "invisible",
    }
    if creation is not None and creation[1] != expected_creation:
        fail("artifact:mesh_proxy:creation:content mismatch")

    parity = _mapping(mesh_proxy.get("rgb_parity"))
    if (
        parity.get("status") != "rgb_parity_pass_pending_shared_capture"
        or parity.get("hide_source_allowed") is not True
        or parity.get("thresholds") != PROXY_RGB_PARITY_THRESHOLDS
        or parity.get("same_physics_step") is not True
        or parity.get("same_renderer_settings") is not True
        or parity.get("parity_capture_isolated") is not True
        or parity.get("source_restored_visible") is not True
        or parity.get("original_capture_visibility")
        != ["inherited", "invisible"]
        or parity.get("proxy_capture_visibility")
        != ["invisible", "inherited"]
        or parity.get("restored_visibility")
        != ["inherited", "invisible"]
        or not _valid_sha256(parity.get("parity_geometry_mesh_sha256"))
        or not _valid_sha256(parity.get("parity_geometry_points_sha256"))
    ):
        fail("artifact:mesh_proxy:parity:receipt contract invalid")
    parity_images: dict[str, tuple[Path, object] | None] = {}
    for label, path_key, hash_key in (
        ("original", "original_rgb_path", "original_rgb_sha256"),
        ("proxy", "proxy_rgb_path", "proxy_rgb_sha256"),
        ("absdiff", "absdiff_path", "absdiff_sha256"),
    ):
        parity_images[label] = checked(
            f"parity_{label}",
            parity.get(path_key),
            parity.get(hash_key),
            lambda path: read_rgb(path, expected_width, expected_height),
        )
    if all(value is not None for value in parity_images.values()):
        original = parity_images["original"][1]
        proxy = parity_images["proxy"][1]
        saved_difference = parity_images["absdiff"][1]
        recomputed = np.abs(
            original.astype(np.int16, copy=False)
            - proxy.astype(np.int16, copy=False)
        ).astype(np.uint8)
        if not np.array_equal(saved_difference, recomputed):
            fail("artifact:mesh_proxy:parity:absdiff pixels differ from recomputation")
        recomputed_metrics = {
            "mean_abs_rgb": float(np.mean(recomputed)),
            "p95_abs_rgb": float(np.percentile(recomputed, 95)),
            "fraction_abs_rgb_gt_16": float(np.mean(recomputed > 16)),
        }
        recorded_metrics = _mapping(parity.get("metrics"))
        for name, value in recomputed_metrics.items():
            recorded = recorded_metrics.get(name)
            if (
                isinstance(recorded, bool)
                or not isinstance(recorded, (int, float))
                or not np.isfinite(float(recorded))
                or not np.isclose(float(recorded), value, rtol=0.0, atol=1e-12)
            ):
                fail(f"artifact:mesh_proxy:parity:{name} differs from pixels")
            threshold = PROXY_RGB_PARITY_THRESHOLDS[
                {
                    "mean_abs_rgb": "mean_abs_rgb_max",
                    "p95_abs_rgb": "p95_abs_rgb_max",
                    "fraction_abs_rgb_gt_16": "fraction_abs_rgb_gt_16_max",
                }[name]
            ]
            if value > threshold:
                fail(f"artifact:mesh_proxy:parity:{name} exceeds threshold")

    render = _mapping(receipt.get("render"))
    frame_count = render.get("frame_count")
    frame_steps = render.get("frame_physics_steps")
    semantics = _mapping(receipt.get("semantics"))
    pairs = semantics.get("pairs")
    refresh_records = mesh_proxy.get("refresh_records")
    stream_only_count = render.get("stream_only_frame_count")
    vial_proxy_count = render.get("vial_proxy_frame_count")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or not isinstance(frame_steps, list)
        or len(frame_steps) != frame_count
        or not isinstance(pairs, list)
        or len(pairs) != frame_count
        or not isinstance(refresh_records, list)
        or isinstance(stream_only_count, bool)
        or not isinstance(stream_only_count, int)
        or stream_only_count < 0
        or isinstance(vial_proxy_count, bool)
        or not isinstance(vial_proxy_count, int)
        or vial_proxy_count <= 0
        or stream_only_count + vial_proxy_count != frame_count
        or len(refresh_records) != vial_proxy_count
        or render.get("vial_proxy_transition_step")
        != frame_steps[stream_only_count]
        or render.get("liquid_representation_monotonic") is not True
    ):
        fail("artifact:mesh_proxy:refresh:frame denominator mismatch")
        return checked_paths, failures
    stream_pairs = [_mapping(pair) for pair in pairs[:stream_only_count]]
    proxy_pairs = [_mapping(pair) for pair in pairs[stream_only_count:]]
    for index, pair in enumerate(stream_pairs):
        phase = pair.get("phase")
        expected_tank = (
            phase.removesuffix("-pour")
            if isinstance(phase, str) and phase.endswith("-pour")
            else None
        )
        if (
            pair.get("liquid_representation") != "stream_only"
            or expected_tank is None
            or pair.get("visible_stream_tanks") != [expected_tank]
            or pair.get("mesh_proxy_refresh_artifact_path") is not None
            or pair.get("mesh_proxy_refresh_artifact_sha256") is not None
            or pair.get("source_isosurface_visibility")
            not in {"absent", "invisible"}
            or pair.get("proxy_visibility") not in {"absent", "invisible"}
            or pair.get("shared_rgb_semantic_capture") is not True
        ):
            fail(f"artifact:mesh_proxy:stream_only[{index}]:receipt mismatch")
    if any(pair.get("liquid_representation") != "vial_proxy" for pair in proxy_pairs):
        fail("artifact:mesh_proxy:refresh:representation downgrade or mismatch")
    first_refresh = _mapping(refresh_records[0])
    first_evidence = _mapping(first_refresh.get("evidence"))
    if (
        parity.get("physics_step") != first_refresh.get("physics_step")
        or parity.get("phase") != first_refresh.get("phase")
        or parity.get("parity_geometry_mesh_sha256")
        != first_evidence.get("mesh_sha256")
        or parity.get("parity_geometry_points_sha256")
        != first_evidence.get("points_sha256")
    ):
        fail("artifact:mesh_proxy:parity:refresh geometry binding mismatch")
    refresh_paths = [
        _mapping(record).get("artifact_path") for record in refresh_records
    ]
    if (
        not all(isinstance(path, str) and path for path in refresh_paths)
        or len(set(path for path in refresh_paths if isinstance(path, str)))
        != vial_proxy_count
    ):
        fail("artifact:mesh_proxy:refresh:artifact paths are not unique")
    for index, raw_record in enumerate(refresh_records):
        record = _mapping(raw_record)
        pair = proxy_pairs[index]
        frame_index = stream_only_count + index
        if (
            record.get("physics_step") != frame_steps[frame_index]
            or record.get("phase") != pair.get("phase")
            or record.get("source_visibility") != "invisible"
            or record.get("proxy_visibility") != "inherited"
            or record.get("shared_rgb_semantic_capture") is not True
            or not _exact_proxy_geometry(
                record.get("evidence"), status="proxy_refreshed_exact"
            )
            or pair.get("mesh_proxy_refresh_artifact_path")
            != record.get("artifact_path")
            or pair.get("mesh_proxy_refresh_artifact_sha256")
            != record.get("artifact_sha256")
            or pair.get("source_isosurface_visibility") != "invisible"
            or pair.get("proxy_visibility") != "inherited"
            or pair.get("shared_rgb_semantic_capture") is not True
            or pair.get("liquid_representation") != "vial_proxy"
        ):
            fail(f"artifact:mesh_proxy:refresh[{index}]:receipt mismatch")
        refresh = checked(
            f"refresh[{index}]",
            record.get("artifact_path"),
            record.get("artifact_sha256"),
            _load_json,
        )
        expected_refresh = {
            key: value
            for key, value in record.items()
            if key not in {"artifact_path", "artifact_sha256"}
        }
        if refresh is not None and refresh[1] != expected_refresh:
            fail(f"artifact:mesh_proxy:refresh[{index}]:content mismatch")

    return checked_paths, failures


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _offline_carrier_occlusion_revalidation(
    app_output: Path,
    receipt: Mapping[str, object],
    *,
    width: int,
    height: int,
    receipt_sha256: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], set[Path]]:
    """Derive, but never rewrite, a source-bound censored-height receipt view."""

    original = dict(receipt)
    arguments = receipt.get("arguments")
    captures = receipt.get("captures")
    if (
        not isinstance(arguments, Mapping)
        or arguments.get("realtime_video") is not True
        or receipt.get("render_path") != COUNT_DRIVEN_RENDER_PATH
        or not isinstance(captures, list)
    ):
        return original, [], set()
    candidates: list[tuple[Mapping[str, object], str]] = []
    for capture in captures:
        if not isinstance(capture, Mapping):
            continue
        gate = capture.get("volume_gate")
        if not isinstance(gate, Mapping):
            continue
        evidence = gate.get("visible_mask_evidence")
        legacy_failure = bool(
            gate.get("state_passed") is True
            and gate.get("visible_passed") is False
            and gate.get("passed") is False
            and gate.get("failure_reasons")
            == ["visible_height_outside_geometry_volume_bounds"]
        )
        current_censored = bool(
            gate.get("visible_passed") is True
            and isinstance(evidence, Mapping)
            and evidence.get("status") == "occlusion_censored"
        )
        if legacy_failure:
            candidates.append((capture, "legacy_failure"))
        elif current_censored:
            candidates.append((capture, "current_censored"))
    if not candidates:
        return original, [], set()

    records: list[dict[str, object]] = []
    checked: set[Path] = set()
    derived = deepcopy(receipt)
    try:
        if (width, height) != (640, 480):
            raise ValueError("FillCam occlusion profile requires 640x480")
        profile = load_fillcam_carrier_occlusion_profile()
        provenance = receipt.get("provenance")
        source_stage_sha256 = (
            provenance.get("requested_stage_sha256")
            if isinstance(provenance, Mapping)
            else None
        )
        scene_contract = receipt.get("scene_contract")
        layer_records = (
            scene_contract.get("source_layer_records")
            if isinstance(scene_contract, Mapping)
            else None
        )
        geometry_records = (
            [
                row
                for row in layer_records
                if isinstance(row, Mapping)
                and str(row.get("resolved_path", "")).endswith(
                    "/geometry/fill_station.usdc"
                )
            ]
            if isinstance(layer_records, list)
            else []
        )
        if len(geometry_records) != 1:
            raise ValueError("source geometry layer binding is not unique")
        source_geometry_sha256 = geometry_records[0].get("sha256")
        occluder_prim_path = str(profile["occluder_prim_path"])
        carrier_profile_record = (
            scene_contract.get("carrier_occlusion_profile")
            if isinstance(scene_contract, Mapping)
            else None
        )
        camera = receipt.get("camera")
        camera_state = (
            camera.get("effective_render_state")
            if isinstance(camera, Mapping)
            else None
        )
        if (
            not isinstance(camera_state, Mapping)
            or camera.get("path") != FILL_CAM_CARRIER_OCCLUSION_CAMERA_PATH
        ):
            raise ValueError("effective FillCam state or path is missing")
        camera_attestation = attest_effective_state_matrix(
            profile["camera_expected_effective_state"],
            camera_state,
            matrix_field="world_transform_matrix",
        )
        if camera_attestation.get("passed") is not True:
            raise ValueError("effective FillCam state does not match profile")
        camera_sha256 = camera_attestation["observed_state_sha256"]
        expected_occluder_state = expected_carrier_occluder_state(profile)
        has_current_censored_candidate = any(
            candidate_kind == "current_censored"
            for _candidate, candidate_kind in candidates
        )
        if has_current_censored_candidate and not isinstance(
            carrier_profile_record, Mapping
        ):
            raise ValueError(
                "current censored receipt lacks composed carrier occluder readback"
            )
        if isinstance(carrier_profile_record, Mapping):
            legacy_expected_effective_state = {
                "prim_path": profile["occluder_prim_path"],
                "time_code": profile["time_code"],
                "mesh_points_count": profile["mesh_points_count"],
                "mesh_face_count": profile["mesh_face_count"],
                "mesh_points_f64_sha256": profile[
                    "mesh_points_f64_sha256"
                ],
                "mesh_face_counts_i64_sha256": profile[
                    "mesh_face_counts_i64_sha256"
                ],
                "mesh_face_indices_i64_sha256": profile[
                    "mesh_face_indices_i64_sha256"
                ],
                "mesh_world_transform_f64_sha256": profile[
                    "mesh_world_transform_f64_sha256"
                ],
                "effective_visibility": profile[
                    "mesh_effective_visibility"
                ],
            }
            effective_state = carrier_profile_record.get(
                "occluder_effective_state"
            )
            saved_occluder_attestation = carrier_profile_record.get(
                "occluder_effective_state_attestation"
            )
            current_record = saved_occluder_attestation is not None
            if current_record:
                occluder_attestation = attest_effective_state_matrix(
                    expected_occluder_state,
                    effective_state,
                    matrix_field="mesh_world_transform_matrix",
                    matrix_raw_hash_field="mesh_world_transform_f64_sha256",
                )
                camera_record_attestation = carrier_profile_record.get(
                    "camera_effective_state_attestation"
                )
                camera_authored_binding = carrier_profile_record.get(
                    "camera_authored_state_binding"
                )
                carrier_record_valid = bool(
                    carrier_profile_record.get("profile_file_sha256")
                    == profile["profile_file_sha256"]
                    and effective_state == occluder_attestation.get("observed_state")
                    and saved_occluder_attestation == occluder_attestation
                    and verify_effective_state_matrix_attestation(
                        saved_occluder_attestation,
                        expected_occluder_state,
                        matrix_field="mesh_world_transform_matrix",
                        matrix_raw_hash_field="mesh_world_transform_f64_sha256",
                    )
                    and carrier_profile_record.get(
                        "occluder_effective_state_sha256"
                    )
                    == occluder_attestation.get("observed_state_sha256")
                    and carrier_profile_record.get("camera_effective_state")
                    == camera_state
                    and carrier_profile_record.get("camera_effective_state_sha256")
                    == camera_sha256
                    and camera_record_attestation == camera_attestation
                    and verify_effective_state_matrix_attestation(
                        camera_record_attestation,
                        profile["camera_expected_effective_state"],
                        matrix_field="world_transform_matrix",
                    )
                    and isinstance(camera_authored_binding, Mapping)
                    and camera_authored_binding.get("passed") is True
                    and camera_authored_binding.get("camera_prim_path")
                    == FILL_CAM_CARRIER_OCCLUSION_CAMERA_PATH
                    and camera_authored_binding.get("source_stage_sha256")
                    == source_stage_sha256
                    and camera_authored_binding.get(
                        "source_geometry_layer_sha256"
                    )
                    == source_geometry_sha256
                    and camera_authored_binding.get(
                        "session_authored_xform_or_projection_properties"
                    )
                    == []
                )
                occluder_binding_basis = (
                    "composed_mesh_pose_visibility_one_ulp_attestation"
                )
            else:
                carrier_record_valid = bool(
                    carrier_profile_record.get("profile_file_sha256")
                    == profile["supersedes_profile_file_sha256"]
                    and effective_state == legacy_expected_effective_state
                    and carrier_profile_record.get(
                        "occluder_effective_state_sha256"
                    )
                    == profile["legacy_occluder_effective_state_sha256"]
                    and _canonical_sha256(effective_state)
                    == profile["legacy_occluder_effective_state_sha256"]
                    and camera_sha256 == profile["camera_effective_state_sha256"]
                )
                occluder_attestation = attest_effective_state_matrix(
                    expected_occluder_state,
                    expected_occluder_state,
                    matrix_field="mesh_world_transform_matrix",
                    matrix_raw_hash_field="mesh_world_transform_f64_sha256",
                )
                occluder_binding_basis = (
                    "legacy_exact_composed_state_plus_frozen_matrix_hash"
                )
            if (
                carrier_profile_record.get("profile_id") != profile["profile_id"]
                or carrier_profile_record.get(
                    "profile_mask_packbits_little_sha256"
                )
                != profile["profile_mask_packbits_little_sha256"]
                or not carrier_record_valid
            ):
                raise ValueError(
                    "effective carrier occluder binding does not match profile"
                )
        else:
            authored_paths = (
                scene_contract.get("authored_paths")
                if isinstance(scene_contract, Mapping)
                else None
            )
            if (
                not isinstance(scene_contract, Mapping)
                or scene_contract.get("source_layers_unchanged") is not True
                or scene_contract.get("frozen_time_code") != 1.0
                or not isinstance(authored_paths, list)
                or any(not isinstance(path, str) for path in authored_paths)
            ):
                raise ValueError(
                    "legacy carrier occluder source/session closure is missing"
                )
            for authored_path in authored_paths:
                authored_prim_path = authored_path.split(".", 1)[0]
                if (
                    authored_prim_path == occluder_prim_path
                    or authored_prim_path.startswith(occluder_prim_path + "/")
                    or occluder_prim_path.startswith(authored_prim_path + "/")
                ):
                    raise ValueError(
                        "legacy session authored the carrier occluder or ancestor"
                    )
            occluder_binding_basis = (
                "exact_source_layers_plus_session_authored_path_exclusion"
            )
            occluder_attestation = attest_effective_state_matrix(
                expected_occluder_state,
                expected_occluder_state,
                matrix_field="mesh_world_transform_matrix",
                matrix_raw_hash_field="mesh_world_transform_f64_sha256",
            )
        semantics = receipt.get("semantics")
        pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
        if not isinstance(pairs, list):
            raise ValueError("semantic capture pairs are missing")

        for candidate, candidate_kind in candidates:
            step = candidate.get("physics_step")
            matching_pairs = [
                pair
                for pair in pairs
                if isinstance(pair, Mapping) and pair.get("physics_step") == step
            ]
            if len(matching_pairs) != 1:
                raise ValueError("occluded stage mask binding is not unique")
            pair = matching_pairs[0]
            if candidate_kind == "legacy_failure":
                historical = profile["measured_cross_check"]
                if (
                    receipt_sha256 != historical["original_receipt_sha256"]
                    or step != 3756
                    or pair.get("mask_sha256")
                    != historical["original_stage2_liquid_mask_sha256"]
                ):
                    raise ValueError(
                        "legacy upgrade is not the approved immutable receipt/stage/mask"
                    )
            candidate_body = candidate.get("render_body_geometry")
            if not isinstance(candidate_body, Mapping):
                raise ValueError("candidate body geometry is missing")
            count_driven_body = receipt.get("count_driven_liquid_body")
            if not isinstance(count_driven_body, Mapping):
                raise ValueError("count-driven body receipt is missing")
            for body_record_key in ("capture_geometry", "stage_geometry"):
                body_records = count_driven_body.get(body_record_key)
                matching_body_records = (
                    [
                        record
                        for record in body_records
                        if isinstance(record, Mapping)
                        and record.get("physics_step") == step
                    ]
                    if isinstance(body_records, list)
                    else []
                )
                if (
                    len(matching_body_records) != 1
                    or matching_body_records[0] != candidate_body
                ):
                    raise ValueError(
                        f"candidate body geometry differs from {body_record_key}"
                    )
            for duplicate_parent_key in ("containment", "point_instancer"):
                duplicate_parent = receipt.get(duplicate_parent_key)
                duplicate_records = (
                    duplicate_parent.get("capture_records")
                    if isinstance(duplicate_parent, Mapping)
                    else None
                )
                matching_duplicates = (
                    [
                        record
                        for record in duplicate_records
                        if isinstance(record, Mapping)
                        and record.get("physics_step") == step
                    ]
                    if isinstance(duplicate_records, list)
                    else []
                )
                if (
                    len(matching_duplicates) != 1
                    or matching_duplicates[0].get("render_body_geometry")
                    != candidate_body
                ):
                    raise ValueError(
                        "candidate body geometry differs from receipt copies"
                    )
            mask_path = resolve_artifact(app_output, pair.get("mask_path"))
            verify_hash(mask_path, pair.get("mask_sha256"))
            mask = read_mask(mask_path, width, height)
            checked.add(mask_path)
            completed = attach_visible_volume_evidence(
                candidate,
                mask,
                center_xy_m=VIAL_CENTER_XY_M,
                bottom_z_m=SOURCE_LIQUID_BOTTOM_M,
                mouth_z_m=VIAL_MOUTH_M,
                circumradius_m=SOURCE_LIQUID_RADIUS_M,
                occlusion_profile=profile,
                source_stage_sha256=source_stage_sha256,
                source_geometry_layer_sha256=source_geometry_sha256,
                camera_effective_state_attestation=camera_attestation,
                occluder_effective_state_attestation=occluder_attestation,
            )
            derived_gate = completed.get("volume_gate")
            if (
                not isinstance(derived_gate, Mapping)
                or not _realtime_visible_height_gate_valid(
                    derived_gate,
                    render_body_geometry=completed.get(
                        "render_body_geometry"
                    ),
                )
                or not isinstance(
                    derived_gate.get("visible_mask_evidence"), Mapping
                )
                or derived_gate["visible_mask_evidence"].get("status")
                != "occlusion_censored"
            ):
                raise ValueError("independent carrier-occlusion proof failed")
            original_gate = candidate["volume_gate"]
            if (
                candidate_kind == "current_censored"
                and current_record
                and (
                    not _realtime_visible_height_gate_valid(
                        original_gate,
                        render_body_geometry=candidate_body,
                    )
                    or _canonical_sha256(derived_gate)
                    != _canonical_sha256(original_gate)
                )
            ):
                raise ValueError(
                    "saved carrier-occlusion proof does not reproduce from mask"
                )

            replacement_count = 0
            for parent_key in ("captures",):
                rows = derived.get(parent_key)
                if isinstance(rows, list):
                    for index, row in enumerate(rows):
                        if (
                            isinstance(row, Mapping)
                            and row.get("physics_step") == step
                            and isinstance(row.get("volume_gate"), Mapping)
                        ):
                            rows[index] = deepcopy(completed)
                            replacement_count += 1
            for parent_key in ("containment", "point_instancer"):
                parent = derived.get(parent_key)
                rows = (
                    parent.get("capture_records")
                    if isinstance(parent, Mapping)
                    else None
                )
                if isinstance(rows, list):
                    for index, row in enumerate(rows):
                        if (
                            isinstance(row, Mapping)
                            and row.get("physics_step") == step
                            and isinstance(row.get("volume_gate"), Mapping)
                        ):
                            rows[index] = deepcopy(completed)
                            replacement_count += 1
            if replacement_count < 2:
                raise ValueError("legacy gate copies were not all found")
            proof = derived_gate["visible_mask_evidence"][
                "occlusion_censor_proof"
            ]
            records.append(
                {
                    "revalidation_type": (
                        "immutable_legacy_carrier_occlusion_height_revalidation"
                        if candidate_kind == "legacy_failure"
                        else (
                            "current_carrier_occlusion_mask_revalidation"
                            if current_record
                            else "legacy_profile_carrier_occlusion_attestation_upgrade"
                        )
                    ),
                    "physics_step": step,
                    "mask_path": pair.get("mask_path"),
                    "mask_sha256": pair.get("mask_sha256"),
                    "original_gate_sha256": _canonical_sha256(
                        candidate["volume_gate"]
                    ),
                    "derived_gate_sha256": _canonical_sha256(derived_gate),
                    "original_receipt_rewritten": False,
                    "occluder_binding_basis": occluder_binding_basis,
                    "passed": True,
                    "proof": proof,
                }
            )
    except (KeyError, TypeError, ValueError, OSError) as error:
        records.append(
            {
                "revalidation_type": (
                    "immutable_legacy_carrier_occlusion_height_revalidation"
                ),
                "original_receipt_rewritten": False,
                "passed": False,
                "failure": f"{type(error).__name__}: {error}",
            }
        )
        return original, records, checked
    return derived, records, checked


def _review_realtime_volume_heights(
    receipt: Mapping[str, object],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """Preserve finite per-stage particle heights as diagnostic warnings."""

    warnings: list[dict[str, object]] = []

    plan = receipt.get("plan")
    pours = plan.get("pours") if isinstance(plan, Mapping) else None
    containment = receipt.get("containment")
    records = (
        containment.get("capture_records")
        if isinstance(containment, Mapping)
        else None
    )
    if (
        not isinstance(pours, list)
        or not pours
        or not all(isinstance(pour, Mapping) for pour in pours)
        or not isinstance(records, list)
    ):
        return warnings, ("realtime_volume_height_record",)

    required_rows = [
        record
        for record in records
        if isinstance(record, Mapping)
        and isinstance(record.get("volume_gate"), Mapping)
        and record["volume_gate"].get("required") is True
    ]
    if len(required_rows) != len(pours):
        return warnings, ("realtime_volume_height_record",)

    cumulative_volume_ml = 0.0
    for pour in pours:
        volume_ml = pour.get("programmed_volume_ml")
        capture_step = pour.get("stage_capture_step")
        if (
            isinstance(volume_ml, bool)
            or not isinstance(volume_ml, (int, float))
            or not np.isfinite(float(volume_ml))
            or float(volume_ml) <= 0.0
            or isinstance(capture_step, bool)
            or not isinstance(capture_step, int)
            or capture_step <= 0
        ):
            return warnings, ("realtime_volume_height_record",)
        cumulative_volume_ml += float(volume_ml)
        matches = [
            record
            for record in required_rows
            if record.get("physics_step") == capture_step
        ]
        if len(matches) != 1:
            return warnings, ("realtime_volume_height_record",)
        gate = matches[0]["volume_gate"]
        observed_volume_ml = gate.get("programmed_cumulative_volume_ml")
        measured_height_m = gate.get("particle_bulk_height_m")
        lower_bound_m = gate.get("height_lower_bound_m")
        upper_bound_m = gate.get("height_upper_bound_m")
        failure_reasons = gate.get("failure_reasons")
        if (
            not isinstance(gate.get("passed"), bool)
            or not isinstance(gate.get("state_passed"), bool)
            or not isinstance(failure_reasons, list)
            or not all(isinstance(reason, str) for reason in failure_reasons)
            or any(
                reason != "particle_height_outside_geometry_volume_bounds"
                for reason in failure_reasons
            )
            or isinstance(observed_volume_ml, bool)
            or not isinstance(observed_volume_ml, (int, float))
            or not np.isfinite(float(observed_volume_ml))
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                for value in (measured_height_m, lower_bound_m, upper_bound_m)
            )
            or float(lower_bound_m) > float(upper_bound_m)
            or not np.isclose(
                float(observed_volume_ml),
                cumulative_volume_ml,
                rtol=0.0,
                atol=1.0e-9,
            )
        ):
            return warnings, ("realtime_volume_height_record",)

        measured = float(measured_height_m)
        lower = float(lower_bound_m)
        upper = float(upper_bound_m)
        if measured < lower:
            direction = "below"
            miss_m = lower - measured
        elif measured > upper:
            direction = "above"
            miss_m = measured - upper
        else:
            direction = "within"
            miss_m = 0.0
        expected_state_passed = direction == "within"
        if gate.get("state_passed") is not expected_state_passed:
            return warnings, ("realtime_volume_height_record",)
        if not expected_state_passed:
            if (
                gate.get("passed") is not False
                or failure_reasons
                != ["particle_height_outside_geometry_volume_bounds"]
            ):
                return warnings, ("realtime_volume_height_record",)
            warnings.append(
                {
                    "warning_type": "particle_height_outside_geometry_volume_bounds",
                    "physics_step": capture_step,
                    "tank_id": pour.get("tank_id"),
                    "programmed_cumulative_volume_ml": float(observed_volume_ml),
                    "measured_particle_height_m": measured,
                    "lower_bound_m": lower,
                    "upper_bound_m": upper,
                    "miss_m": round(float(miss_m), 12),
                    "direction": direction,
                    "original_gate_passed": gate.get("passed"),
                    "original_state_passed": gate.get("state_passed"),
                    "original_failure_reasons": list(failure_reasons),
                    "hard_gate": False,
                    "claim": "diagnostic_warning_not_a_physics_pass",
                }
            )
        elif gate.get("passed") is not True or failure_reasons != []:
            return warnings, ("realtime_volume_height_record",)
    return warnings, ()


def validate_output(app_output: Path) -> dict[str, object]:
    failures: list[str] = []
    warnings: list[dict[str, object]] = []
    checked_paths: set[Path] = set()

    def fail(message: str) -> None:
        if message not in failures:
            failures.append(message)

    try:
        receipt_path = resolve_artifact(app_output, "receipt.json")
        receipt_value = _load_json(receipt_path)
    except (ValueError, ArtifactValidationError):
        return {
            "status": "insufficient_evidence",
            "verified_artifact_count": 0,
            "failures": ["receipt_missing_or_unreadable"],
            "warnings": [],
        }
    if not isinstance(receipt_value, Mapping):
        return {
            "status": "insufficient_evidence",
            "verified_artifact_count": 0,
            "failures": ["receipt_missing_or_unreadable"],
            "warnings": [],
        }
    receipt = receipt_value
    checked_paths.add(receipt_path)
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    preliminary_render = _mapping(receipt.get("render"))
    preliminary_resolution = preliminary_render.get("resolution")
    offline_revalidations: list[dict[str, object]] = []
    if (
        isinstance(preliminary_resolution, list)
        and len(preliminary_resolution) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in preliminary_resolution
        )
    ):
        receipt, offline_revalidations, revalidated_paths = (
            _offline_carrier_occlusion_revalidation(
                app_output,
                receipt,
                width=int(preliminary_resolution[0]),
                height=int(preliminary_resolution[1]),
                receipt_sha256=receipt_sha256,
            )
        )
        checked_paths.update(revalidated_paths)
        if any(
            not isinstance(record, Mapping) or record.get("passed") is not True
            for record in offline_revalidations
        ):
            fail("receipt:carrier_occlusion_offline_revalidation")
    try:
        arguments = _mapping(receipt.get("arguments"))
        if arguments.get("realtime_video") is True:
            receipt_validator = _validate_realtime_height_banded_receipt
        else:
            receipt_validator = (
                validate_paint_ball_receipt
                if receipt.get("render_path")
                in {EMITTER_RENDER_PATH, COUNT_DRIVEN_RENDER_PATH}
                else validate_three_pour_receipt
            )
        for reason in receipt_validator(receipt):
            fail(f"receipt:{reason}")
        if arguments.get("realtime_video") is True:
            height_warnings, height_failures = _review_realtime_volume_heights(
                receipt
            )
            warnings.extend(height_warnings)
            for reason in height_failures:
                fail(f"receipt:{reason}")
    except Exception as exc:  # the offline checker must fail closed on malformed receipts
        fail(f"receipt_validator_error:{type(exc).__name__}")

    render = _mapping(receipt.get("render"))
    resolution = render.get("resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in resolution)
        or resolution[0] <= 0
        or resolution[1] <= 0
    ):
        fail("artifact_resolution_invalid")
        width, height = 640, 480
    else:
        width, height = int(resolution[0]), int(resolution[1])

    verified_hashes: dict[Path, str] = {}

    def artifact(
        label: str,
        relative_path: object,
        expected_hash: object,
        reader: Callable[[Path], object] | None = None,
    ) -> tuple[Path, object | None] | None:
        try:
            path = resolve_artifact(app_output, relative_path)
            if path in verified_hashes and verified_hashes[path] != expected_hash:
                raise ArtifactValidationError(
                    f"conflicting receipt hashes for {path.name}"
                )
            verify_hash(path, expected_hash)
            verified_hashes[path] = str(expected_hash)
            checked_paths.add(path)
            value = reader(path) if reader is not None else None
            return path, value
        except (ValueError, ArtifactValidationError, OSError) as exc:
            fail(f"artifact:{label}:{exc}")
            return None

    def artifact_list(
        label: str,
        paths: object,
        hashes: object,
        reader: Callable[[Path], object] | None = None,
    ) -> list[tuple[Path, object | None] | None]:
        if not isinstance(paths, list):
            return []
        hash_list = hashes if isinstance(hashes, list) else []
        return [
            artifact(
                f"{label}[{index}]",
                relative_path,
                hash_list[index] if index < len(hash_list) else None,
                reader,
            )
            for index, relative_path in enumerate(paths)
        ]

    rgb_reader = lambda path: read_rgb(path, width, height)
    mask_reader = lambda path: read_mask(path, width, height)
    semantic_reader = lambda path: read_semantic(path, width, height)

    particle_system = _mapping(receipt.get("particle_system"))
    runtime_configuration = particle_system.get("runtime_configuration")
    if isinstance(runtime_configuration, Mapping):
        captures = receipt.get("captures")
        if not isinstance(captures, list) or not captures:
            fail("artifact:particle_snapshot:captures missing")
        else:
            for capture_index, capture_value in enumerate(captures):
                capture = _mapping(capture_value)
                snapshot = _mapping(capture.get("particle_snapshot"))
                arrays = snapshot.get("arrays")
                if (
                    snapshot.get("arrays_saved_before_assertions") is not True
                    or not isinstance(arrays, Mapping)
                    or set(arrays) != RUNTIME_PARTICLE_ARRAYS
                ):
                    fail(f"artifact:particle_snapshot[{capture_index}]:manifest missing")
                    continue
                artifact(
                    f"particle_snapshot_manifest[{capture_index}]",
                    snapshot.get("snapshot_path"),
                    snapshot.get("snapshot_sha256"),
                    _load_json,
                )
                expected_rows = None
                for name, record_value in arrays.items():
                    record = _mapping(record_value)
                    checked = artifact(
                        f"particle_snapshot[{capture_index}].{name}",
                        record.get("path"),
                        record.get("sha256"),
                        read_particle_array,
                    )
                    if checked is None:
                        continue
                    values = checked[1]
                    if (
                        record.get("shape") != list(values.shape)
                        or record.get("dtype") != str(values.dtype)
                        or record.get("finite") is not True
                    ):
                        fail(
                            f"artifact:particle_snapshot[{capture_index}].{name}:"
                            "receipt mismatch"
                        )
                    rows = len(values) if values.ndim else -1
                    if expected_rows is None:
                        expected_rows = rows
                    elif rows != expected_rows:
                        fail(
                            f"artifact:particle_snapshot[{capture_index}]:row mismatch"
                        )

    if receipt.get("render_path") not in {
        EMITTER_RENDER_PATH,
        COUNT_DRIVEN_RENDER_PATH,
    }:
        proxy_checked_paths, proxy_failures = validate_mesh_proxy_artifacts(
            app_output,
            receipt,
            expected_width=width,
            expected_height=height,
        )
        checked_paths.update(proxy_checked_paths)
        for proxy_failure in proxy_failures:
            fail(proxy_failure)

    artifact_list(
        "frame",
        render.get("frame_paths"),
        render.get("frame_sha256"),
        rgb_reader,
    )
    artifact_list(
        "stage_frame",
        render.get("stage_frame_paths"),
        render.get("stage_frame_sha256"),
        rgb_reader,
    )

    parity = _mapping(receipt.get("pre_pour_parity"))
    for label, path_key, hash_key in (
        ("parity_reference", "reference_path", "reference_sha256"),
        ("parity_prefill", "f2_prefill_path", "f2_prefill_sha256"),
        ("parity_absdiff", "absdiff_path", "absdiff_sha256"),
    ):
        if parity.get(path_key) is not None or parity.get(hash_key) is not None:
            artifact(label, parity.get(path_key), parity.get(hash_key), rgb_reader)

    semantics = _mapping(receipt.get("semantics"))
    pairs = semantics.get("pairs")
    if isinstance(pairs, list):
        for index, pair_value in enumerate(pairs):
            pair = _mapping(pair_value)
            artifact(
                f"semantic_pair_rgb[{index}]",
                pair.get("rgb_path"),
                pair.get("rgb_sha256"),
                rgb_reader,
            )
            checked_mask = artifact(
                f"liquid_mask[{index}]",
                pair.get("mask_path"),
                pair.get("mask_sha256"),
                mask_reader,
            )
            checked_semantic = artifact(
                f"semantic_array[{index}]",
                pair.get("semantic_path"),
                pair.get("semantic_sha256"),
                semantic_reader,
            )
            checked_instance = None
            checked_instance_mapping = None
            if receipt.get("render_path") == COUNT_DRIVEN_RENDER_PATH:
                checked_instance = artifact(
                    f"instance_array[{index}]",
                    pair.get("instance_path"),
                    pair.get("instance_sha256"),
                    semantic_reader,
                )
                checked_instance_mapping = artifact(
                    f"instance_mapping[{index}]",
                    pair.get("instance_mapping_path"),
                    pair.get("instance_mapping_sha256"),
                    _load_json,
                )
            if checked_mask is not None:
                _path, mask_value = checked_mask
                observed_pixels = int(np.count_nonzero(mask_value))
                if observed_pixels != pair.get("liquid_pixel_count"):
                    fail(
                        f"artifact:liquid_mask[{index}]:liquid pixel count "
                        f"{observed_pixels} != {pair.get('liquid_pixel_count')!r}"
                    )
            if checked_mask is not None and checked_semantic is not None:
                try:
                    validate_semantic_mask_evidence(
                        checked_semantic[1], checked_mask[1], pair
                    )
                except ArtifactValidationError as exc:
                    fail(f"artifact:semantic_mask[{index}]:{exc}")
            if (
                checked_mask is not None
                and checked_instance is not None
                and checked_instance_mapping is not None
            ):
                mapping = checked_instance_mapping[1]
                if mapping != pair.get("instance_id_to_prim_path"):
                    fail(
                        f"artifact:instance_mapping[{index}]:"
                        "mapping differs from receipt"
                    )
                try:
                    overlap = exact_instance_semantic_overlap(
                        checked_instance[1],
                        mapping,
                        checked_mask[1],
                        instance_path=PERSISTENT_RENDER_STREAM_PATH,
                        expected_visible=bool(pair.get("visible_stream_tanks")),
                    )
                except LabelMappingError as exc:
                    fail(f"artifact:stream_overlap[{index}]:{exc}")
                else:
                    observed_overlap = {
                        "instance_path": overlap.instance_path,
                        "expected_visible": overlap.expected_visible,
                        "instance_ids": list(overlap.instance_ids),
                        "instance_pixel_count": overlap.instance_pixel_count,
                        "liquid_overlap_pixel_count": (
                            overlap.liquid_overlap_pixel_count
                        ),
                        "non_liquid_instance_pixel_count": (
                            overlap.non_liquid_instance_pixel_count
                        ),
                        "coverage_fraction": overlap.coverage_fraction,
                        "passed": overlap.passed,
                    }
                    if observed_overlap != pair.get("stream_semantic_overlap"):
                        fail(
                            f"artifact:stream_overlap[{index}]:"
                            "pixel evidence differs from receipt"
                        )
    artifact_list(
        "semantic_array_manifest",
        semantics.get("frame_semantic_paths"),
        semantics.get("frame_semantic_sha256"),
        semantic_reader,
    )

    frame_count = render.get("frame_count")
    if render.get("video_path") is not None or render.get("video_sha256") is not None:
        video_result = artifact(
            "video",
            render.get("video_path"),
            render.get("video_sha256"),
        )
        if video_result is not None:
            video_frame_count = render.get("video_decoded_frame_count")
            repeat_counts = render.get("video_frame_repeat_counts")
            repeat_denominator_valid = True
            if repeat_counts is not None:
                if _mapping(receipt.get("arguments")).get("realtime_video") is True:
                    repeat_denominator_valid = bool(
                        isinstance(video_frame_count, int)
                        and not isinstance(video_frame_count, bool)
                        and isinstance(repeat_counts, list)
                        and len(repeat_counts) == video_frame_count
                        and repeat_counts == [1] * video_frame_count
                    )
                else:
                    repeat_denominator_valid = bool(
                        isinstance(frame_count, int)
                        and not isinstance(frame_count, bool)
                        and isinstance(repeat_counts, list)
                        and len(repeat_counts) == frame_count
                        and all(
                            isinstance(value, int)
                            and not isinstance(value, bool)
                            and value > 0
                            for value in repeat_counts
                        )
                        and sum(repeat_counts) == video_frame_count
                    )
            if (
                isinstance(video_frame_count, int)
                and not isinstance(video_frame_count, bool)
                and video_frame_count > 0
                and repeat_denominator_valid
            ):
                try:
                    decoded = validate_video(
                        video_result[0], width, height, video_frame_count
                    )
                    if decoded != video_frame_count:
                        fail(
                            "artifact:video:decoded frame count differs from receipt"
                        )
                except ArtifactValidationError as exc:
                    fail(f"artifact:video:{exc}")
            else:
                fail("artifact:video:encoded frame denominator is invalid")

    stream_render = _mapping(receipt.get("stream_render"))
    stream_records = stream_render.get("records")
    if isinstance(stream_records, list):
        for index, value in enumerate(stream_records):
            record = _mapping(value)
            checked = artifact(
                f"stream_state[{index}]",
                record.get("state_artifact_path"),
                record.get("state_artifact_sha256"),
                _load_json,
            )
            if checked is None:
                continue
            state = _mapping(checked[1])
            for key in (
                "ordinal",
                "tank_id",
                "start_step",
                "end_step",
                "transmission_color",
                "active_only_during_own_pour",
            ):
                if state.get(key) != record.get(key):
                    fail(f"artifact:stream_state[{index}]:content mismatch for {key}")

    global_blend = _mapping(receipt.get("global_blend"))
    stage_states = global_blend.get("stage_states")
    if isinstance(stage_states, list):
        for index, value in enumerate(stage_states):
            record = _mapping(value)
            checked = artifact(
                f"blend_stage_state[{index}]",
                record.get("state_artifact_path"),
                record.get("state_artifact_sha256"),
                _load_json,
            )
            if checked is None:
                continue
            state = _mapping(checked[1])
            for key in (
                "ordinal",
                "tank_id",
                "physics_step",
                "cumulative_particle_counts",
                "cumulative_particle_count",
                "component_fractions",
                "transmission_color",
                "usd_readback_transmission_color",
            ):
                if state.get(key) != record.get(key):
                    fail(
                        f"artifact:blend_stage_state[{index}]:content mismatch for {key}"
                    )

    trace_checked = None
    if (
        global_blend.get("emission_trace_path") is not None
        or global_blend.get("emission_trace_sha256") is not None
    ):
        trace_checked = artifact(
            "blend_trace",
            global_blend.get("emission_trace_path"),
            global_blend.get("emission_trace_sha256"),
            _load_json,
        )
    if trace_checked is not None:
        trace = _mapping(trace_checked[1])
        events = trace.get("events")
        expected_events = global_blend.get("emission_trace_event_count")
        if (
            trace.get("event_count") != expected_events
            or not isinstance(events, list)
            or len(events) != expected_events
        ):
            fail("artifact:blend_trace:event count mismatch")
        elif isinstance(events, list):
            for index, event_value in enumerate(events, start=1):
                event = _mapping(event_value)
                requested = event.get("transmission_color")
                readback = event.get("usd_readback_transmission_color")
                if (
                    event.get("update_ordinal") != index
                    or not isinstance(requested, list)
                    or not isinstance(readback, list)
                    or len(requested) != 3
                    or len(readback) != 3
                    or not np.allclose(
                        requested, readback, rtol=0.0, atol=1e-7
                    )
                ):
                    fail(f"artifact:blend_trace:event {index} readback mismatch")

    return {
        "status": "pass" if not failures else "insufficient_evidence",
        "verified_artifact_count": len(checked_paths),
        "failures": failures,
        "warnings": warnings,
        "offline_revalidations": offline_revalidations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "app_output",
        type=Path,
        help="copied app-output directory containing receipt.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_output(args.app_output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate copied-back schema-14 F2 artifacts against their receipt."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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

from fluidmix.isosurface_mesh_proxy import PROXY_RGB_PARITY_THRESHOLDS
from fluidmix.physx_three_pour import (
    FLUIDMIX_NAMESPACE,
    GENERATED_ISOSURFACE_PATH,
    LIQUID_PROXY_PATH,
    validate_three_pour_receipt,
)
from fluidmix.render_capture import LabelMappingError, exact_semantic_class_mask


class ArtifactValidationError(ValueError):
    """One copied artifact is missing, unsafe, corrupt, or inconsistent."""


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
    if not isinstance(stage_surface_sync_records, list) or len(
        stage_surface_sync_records
    ) > 1:
        fail("artifact:mesh_proxy:stage_surface_sync:receipt contract invalid")
        stage_surface_sync_records = []
    for index, raw_record in enumerate(stage_surface_sync_records):
        record = _mapping(raw_record)
        sync = checked(
            f"stage_surface_sync[{index}]",
            record.get("artifact_path"),
            record.get("artifact_sha256"),
            _load_json,
        )
        expected_sync = {
            key: value
            for key, value in record.items()
            if key not in {"artifact_path", "artifact_sha256"}
        }
        if sync is not None and sync[1] != expected_sync:
            fail(
                f"artifact:mesh_proxy:stage_surface_sync[{index}]:content mismatch"
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


def validate_output(app_output: Path) -> dict[str, object]:
    failures: list[str] = []
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
        }
    if not isinstance(receipt_value, Mapping):
        return {
            "status": "insufficient_evidence",
            "verified_artifact_count": 0,
            "failures": ["receipt_missing_or_unreadable"],
        }
    receipt = receipt_value
    checked_paths.add(receipt_path)
    try:
        for reason in validate_three_pour_receipt(receipt):
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
            if isinstance(frame_count, int) and not isinstance(frame_count, bool):
                try:
                    decoded = validate_video(
                        video_result[0], width, height, frame_count
                    )
                    if decoded != render.get("video_decoded_frame_count"):
                        fail(
                            "artifact:video:decoded frame count differs from receipt"
                        )
                except ArtifactValidationError as exc:
                    fail(f"artifact:video:{exc}")
            else:
                fail("artifact:video:receipt frame count is invalid")

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


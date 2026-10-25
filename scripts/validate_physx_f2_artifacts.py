#!/usr/bin/env python3
"""Validate copied-back schema-8 F2 artifacts against their receipt."""

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

from fluidmix.physx_three_pour import validate_three_pour_receipt


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


def read_instance(path: Path, expected_width: int, expected_height: int):
    try:
        instance = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ArtifactValidationError(f"instance array is unreadable: {path.name}") from exc
    expected_shape = (expected_height, expected_width)
    if instance.shape != expected_shape:
        raise ArtifactValidationError(
            f"instance shape mismatch for {path.name}: {instance.shape}"
        )
    if np.issubdtype(instance.dtype, np.bool_) or not np.issubdtype(
        instance.dtype, np.integer
    ):
        raise ArtifactValidationError(
            f"instance dtype is not integer for {path.name}: {instance.dtype}"
        )
    if np.any(instance < 0):
        raise ArtifactValidationError(f"instance IDs are negative: {path.name}")
    return instance


def validate_instance_mask_evidence(
    instance: np.ndarray,
    mask: np.ndarray,
    pair: Mapping[str, Any],
    semantics: Mapping[str, Any],
) -> None:
    """Prove a saved liquid mask is exactly the reviewed renderer-ID union."""

    selected_ids = pair.get("selected_instance_ids")
    id_to_path = pair.get("selected_instance_id_to_path")
    selected_paths = pair.get("selected_instance_paths")
    isosurface_path = semantics.get("required_isosurface_path")
    stream_paths = semantics.get("allowed_stream_paths")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in selected_ids
        )
        or not isinstance(id_to_path, Mapping)
        or set(id_to_path) != {str(value) for value in selected_ids}
        or not isinstance(selected_paths, list)
        or not isinstance(isosurface_path, str)
        or not isinstance(stream_paths, list)
        or not all(isinstance(path, str) for path in stream_paths)
    ):
        raise ArtifactValidationError("instance-path receipt mapping is malformed")
    allowed = {isosurface_path, *stream_paths}
    if (
        any(not isinstance(path, str) or path not in allowed for path in id_to_path.values())
        or set(id_to_path.values()) != set(selected_paths)
    ):
        raise ArtifactValidationError("instance-path receipt mapping is unreviewed")

    expected_mask = np.isin(instance, selected_ids)
    if not np.array_equal(mask > 0, expected_mask):
        raise ArtifactValidationError("mask differs from exact instance IDs")

    iso_ids = [
        value for value in selected_ids if id_to_path[str(value)] == isosurface_path
    ]
    stream_ids = [
        value for value in selected_ids if id_to_path[str(value)] in stream_paths
    ]
    observed_iso = int(np.count_nonzero(np.isin(instance, iso_ids)))
    observed_stream = int(np.count_nonzero(np.isin(instance, stream_ids)))
    observed_total = int(np.count_nonzero(expected_mask))
    if pair.get("isosurface_pixel_count") != observed_iso:
        raise ArtifactValidationError(
            "isosurface pixel count differs from instance artifact"
        )
    if pair.get("stream_pixel_count") != observed_stream:
        raise ArtifactValidationError("stream pixel count differs from instance artifact")
    if pair.get("liquid_pixel_count") != observed_total:
        raise ArtifactValidationError("liquid pixel count differs from instance artifact")


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
    instance_reader = lambda path: read_instance(path, width, height)

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
            checked_instance = artifact(
                f"instance_array[{index}]",
                pair.get("instance_path"),
                pair.get("instance_sha256"),
                instance_reader,
            )
            if checked_mask is not None:
                _path, mask_value = checked_mask
                observed_pixels = int(np.count_nonzero(mask_value))
                if observed_pixels != pair.get("liquid_pixel_count"):
                    fail(
                        f"artifact:liquid_mask[{index}]:liquid pixel count "
                        f"{observed_pixels} != {pair.get('liquid_pixel_count')!r}"
                    )
            if checked_mask is not None and checked_instance is not None:
                try:
                    validate_instance_mask_evidence(
                        checked_instance[1], checked_mask[1], pair, semantics
                    )
                except ArtifactValidationError as exc:
                    fail(f"artifact:instance_mask[{index}]:{exc}")
    artifact_list(
        "instance_array_manifest",
        semantics.get("frame_instance_paths"),
        semantics.get("frame_instance_sha256"),
        instance_reader,
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


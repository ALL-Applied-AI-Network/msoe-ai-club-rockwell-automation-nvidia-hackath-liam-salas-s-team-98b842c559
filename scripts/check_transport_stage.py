#!/usr/bin/env python3
"""Independently check rendered transport states, endpoint substitution, and seams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.rigid_vial_transport import (  # noqa: E402
    INCOMING_PRIM_PATHS,
    INCOMING_VIAL_PATH,
    OUTGOING_PRIM_PATHS,
    QUEUED_PRIM_PATHS,
)
from fluidmix.transport_stage import (  # noqa: E402
    ALL_STATE_INDICES,
    BETWEEN_MODE,
    capture_transport_implementation,
    ENTRY_MODE,
    EXIT_MODE,
    package_state_indices,
    validate_pilot_visual_review,
    validate_runtime_partition,
    validate_wp1_fill_readiness,
)


PILOT_STATE_INDICES = (0, 6, 12)
FRAME_SHAPE = (480, 640, 3)
RTX_RENDER_MODE_SETTING = "/rtx/rendermode"
RTX_RENDER_MODE_TARGET = "RaytracedLighting"
RTX_RENDER_MODE_MAX_APP_UPDATES = 120
FROZEN_FILL_ANCHOR_X_PX = 304.5
BACKGROUND_SENTINELS = (
    (25, 25),
    (100, 80),
    (210, 90),
    (430, 80),
    (610, 25),
    (30, 450),
    (610, 450),
)
GROUP_PREFIXES = {
    "outgoing": tuple(sorted(OUTGOING_PRIM_PATHS)),
    "incoming": tuple(sorted(INCOMING_PRIM_PATHS)),
    "queued": tuple(sorted(QUEUED_PRIM_PATHS)),
    "incoming_labels": (
        "/FillStation/FillStation_LocalRoot/FluidmixTransport/IncomingLabels",
    ),
    "queue_tail": (
        "/FillStation/FillStation_LocalRoot/FluidmixTransport/QueueTail",
    ),
}


def _validate_render_mode_transitions(
    value: object, *, expected_state_indices: Sequence[int]
) -> list[dict[str, Any]]:
    """Validate the runtime opening provenance recorded by the renderer."""

    required = {
        "state_index",
        "opening_kind",
        "setting_path",
        "requested",
        "before",
        "actual_readback",
        "app_updates",
        "max_app_updates",
        "passed",
    }
    expected_openings = [
        *(("preflight", index) for index in expected_state_indices),
        *(("capture", index) for index in expected_state_indices),
    ]
    if not expected_openings or not isinstance(value, list) or len(value) != len(
        expected_openings
    ):
        raise ValueError("transport render-mode transition membership is invalid")
    result: list[dict[str, Any]] = []
    for row, (opening_kind, state_index) in zip(value, expected_openings, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != required
            or type(row.get("state_index")) is not int
            or row.get("state_index") != state_index
            or row.get("opening_kind") != opening_kind
            or row.get("setting_path") != RTX_RENDER_MODE_SETTING
            or row.get("requested") != RTX_RENDER_MODE_TARGET
            or not isinstance(row.get("before"), str)
            or row.get("actual_readback") != RTX_RENDER_MODE_TARGET
            or type(row.get("app_updates")) is not int
            or not 0 <= row["app_updates"] <= RTX_RENDER_MODE_MAX_APP_UPDATES
            or row.get("max_app_updates") != RTX_RENDER_MODE_MAX_APP_UPDATES
            or row.get("passed") is not True
        ):
            raise ValueError("transport render-mode transition evidence is invalid")
        result.append(dict(row))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_evidence_closure_shape(value: object) -> dict[str, Any]:
    required = {
        "source_files",
        "snapshot_arrays",
        "state_layers",
        "dependency_files",
        "counters",
        "sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("transport evidence closure schema is malformed")
    result = dict(value)
    for key, expected_count in (
        ("source_files", 2),
        ("snapshot_arrays", 13),
        ("state_layers", 13),
        ("dependency_files", None),
    ):
        rows = result.get(key)
        if (
            not isinstance(rows, list)
            or (expected_count is None and not rows)
            or (expected_count is not None and len(rows) != expected_count)
        ):
            raise ValueError(f"transport {key} closure membership is invalid")
        for row in rows:
            if (
                not isinstance(row, Mapping)
                or set(row) != {"path", "bytes", "sha256"}
                or not isinstance(row.get("path"), str)
                or not row.get("path")
                or type(row.get("bytes")) is not int
                or row.get("bytes", -1) < 0
                or not _valid_sha256(row.get("sha256"))
            ):
                raise ValueError(f"transport {key} closure record is malformed")
    if result.get("counters") != {
        "physics_steps": 0,
        "physics_callbacks": 0,
        "timeline_playing": False,
        "source_time_code": 1,
    }:
        raise ValueError("transport evidence closure counters are invalid")
    payload = {key: result[key] for key in result if key != "sha256"}
    observed = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    if not _valid_sha256(result.get("sha256")) or result["sha256"] != observed:
        raise ValueError("transport evidence closure digest is invalid")
    return result


def _strict_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value


def _bound_output_path(output: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise ValueError(f"{label} path is invalid")
    path = (output / raw).resolve()
    try:
        path.relative_to(output)
    except ValueError as error:
        raise ValueError(f"{label} path escapes the candidate output") from error
    return path


def _load_rgb(path: Path, expected_hash: object, label: str) -> np.ndarray:
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError(f"{label} differs from its receipt")
    value = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if value is None or value.shape != FRAME_SHAPE:
        raise ValueError(f"{label} is not decodable 640x480 RGB")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path, expected_hash: object, label: str) -> np.ndarray:
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError(f"{label} differs from its receipt")
    value = cv2.imread(os.fspath(path), cv2.IMREAD_GRAYSCALE)
    if (
        value is None
        or value.shape != FRAME_SHAPE[:2]
        or not set(np.unique(value)).issubset({0, 255})
    ):
        raise ValueError(f"{label} is not a binary 640x480 mask")
    return value


def _decode_video(path: Path, expected_hash: object, label: str) -> list[np.ndarray]:
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError(f"{label} differs from its accepted receipt")
    capture = cv2.VideoCapture(os.fspath(path))
    if not capture.isOpened():
        raise ValueError(f"{label} is not decodable")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != FRAME_SHAPE:
                raise ValueError(f"{label} has a non-640x480 frame")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"{label} has no frames")
    return frames


def _mad(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def _matched_p99(frames: Sequence[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    return float(np.percentile([_mad(left, right) for left, right in zip(frames, frames[1:])], 99))


def _vial_center_x(
    instance: np.ndarray, mapping: Mapping[str, str], path_prefix: str
) -> float | None:
    ids = [
        int(instance_id)
        for instance_id, path in mapping.items()
        if path == path_prefix or path.startswith(path_prefix + "/")
    ]
    pixels = np.argwhere(np.isin(instance, ids))
    return None if not len(pixels) else float(np.mean(pixels[:, 1]))


def _mode_occupancy_evidence(
    mode: str, captures: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Summarize mode occupancy from independently reloaded instance evidence."""

    if mode not in {ENTRY_MODE, BETWEEN_MODE, EXIT_MODE} or not captures:
        raise ValueError("transport mode occupancy inputs are invalid")
    ordered = sorted(captures, key=lambda row: int(row["state_index"]))
    indices = [int(row["state_index"]) for row in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError("transport mode occupancy state indices are duplicated")
    early_index = indices[0]
    late_indices = [11, 12] if {11, 12}.issubset(indices) else [indices[-1]]
    pixel_counts: dict[str, dict[str, int]] = {}
    mapped_counts: dict[str, dict[str, int]] = {}
    incoming_centers: dict[str, float | None] = {}
    for row in ordered:
        state_index = int(row["state_index"])
        instance = row.get("instance")
        mapping = row.get("mapping")
        masks = row.get("masks")
        if (
            not isinstance(instance, np.ndarray)
            or instance.shape != FRAME_SHAPE[:2]
            or not isinstance(mapping, Mapping)
            or not isinstance(masks, Mapping)
            or set(masks) != set(GROUP_PREFIXES)
        ):
            raise ValueError("transport mode occupancy capture is malformed")
        state_pixels: dict[str, int] = {}
        state_mapped: dict[str, int] = {}
        for name, prefixes in GROUP_PREFIXES.items():
            ids = [
                int(instance_id)
                for instance_id, path in mapping.items()
                if any(
                    path == prefix or path.startswith(prefix + "/")
                    for prefix in prefixes
                )
            ]
            expected = np.isin(instance, ids).astype(np.uint8) * 255
            mask = masks.get(name)
            if not isinstance(mask, np.ndarray) or not np.array_equal(mask, expected):
                raise ValueError(
                    f"state {state_index} {name} group mask differs from instance evidence"
                )
            state_pixels[name] = int(np.count_nonzero(expected))
            state_mapped[name] = len(ids)
        key = str(state_index)
        pixel_counts[key] = state_pixels
        mapped_counts[key] = state_mapped
        incoming_centers[key] = _vial_center_x(
            instance, mapping, INCOMING_VIAL_PATH
        )

    early = pixel_counts[str(early_index)]
    late = [pixel_counts[str(index)] for index in late_indices]
    late_centers = [incoming_centers[str(index)] for index in late_indices]
    outgoing_absent_all = all(
        counts["outgoing"] == 0 for counts in pixel_counts.values()
    )
    outgoing_present_early = early["outgoing"] > 0
    outgoing_absent_late = all(counts["outgoing"] == 0 for counts in late)
    incoming_present_late = all(counts["incoming"] > 0 for counts in late)
    incoming_centered_late = all(
        center is not None and abs(center - FROZEN_FILL_ANCHOR_X_PX) <= 3.0
        for center in late_centers
    )
    exit_new_groups_absent_all = all(
        counts["incoming_labels"] == 0 and counts["queue_tail"] == 0
        for counts in pixel_counts.values()
    ) and all(
        counts["incoming_labels"] == 0 and counts["queue_tail"] == 0
        for counts in mapped_counts.values()
    )

    checks = {
        "entry_outgoing_absent_all": outgoing_absent_all if mode == ENTRY_MODE else None,
        "outgoing_present_early": (
            outgoing_present_early if mode in {BETWEEN_MODE, EXIT_MODE} else None
        ),
        "outgoing_absent_late": (
            outgoing_absent_late if mode in {BETWEEN_MODE, EXIT_MODE} else None
        ),
        "incoming_present_late": (
            incoming_present_late if mode in {ENTRY_MODE, BETWEEN_MODE} else None
        ),
        "incoming_centered_late": (
            incoming_centered_late if mode in {ENTRY_MODE, BETWEEN_MODE} else None
        ),
        "exit_new_groups_absent_all": (
            exit_new_groups_absent_all if mode == EXIT_MODE else None
        ),
    }
    applicable = [value for value in checks.values() if value is not None]
    return {
        "mode": mode,
        "state_indices": indices,
        "early_state_index": early_index,
        "late_state_indices": late_indices,
        "pixel_counts_by_state": pixel_counts,
        "mapped_instance_counts_by_state": mapped_counts,
        "incoming_center_x_px_by_state": incoming_centers,
        **checks,
        "mode_occupancy_valid": bool(applicable)
        and all(value is True for value in applicable),
    }


def _load_capture(
    output: Path, row: Mapping[str, Any]
) -> dict[str, Any]:
    index = row.get("state_index")
    if type(index) is not int:
        raise ValueError("transport capture state index is invalid")
    rgb = _load_rgb(
        _bound_output_path(output, row.get("rgb_path"), f"state {index} RGB"),
        row.get("rgb_sha256"),
        f"state {index} RGB",
    )
    instance_path = _bound_output_path(
        output, row.get("instance_path"), f"state {index} instance raster"
    )
    if not instance_path.is_file() or _sha256(instance_path) != row.get("instance_sha256"):
        raise ValueError(f"state {index} instance raster differs from its receipt")
    instance = np.load(instance_path, allow_pickle=False)
    if instance.shape != FRAME_SHAPE[:2] or instance.dtype != np.uint32:
        raise ValueError(f"state {index} instance raster is malformed")
    mapping_path = _bound_output_path(
        output, row.get("mapping_path"), f"state {index} mapping"
    )
    mapping = _strict_object(mapping_path, f"state {index} mapping")
    if _sha256(mapping_path) != row.get("mapping_sha256"):
        raise ValueError(f"state {index} instance mapping differs from its receipt")
    if any(not key.isdigit() or not isinstance(value, str) for key, value in mapping.items()):
        raise ValueError(f"state {index} instance mapping is malformed")
    observed_ids = {int(value) for value in np.unique(instance)} - {0}
    if not observed_ids.issubset({int(key) for key in mapping}):
        raise ValueError(f"state {index} instance raster has an unmapped ID")
    mask_rows = row.get("group_masks")
    if not isinstance(mask_rows, Mapping) or set(mask_rows) != set(GROUP_PREFIXES):
        raise ValueError(f"state {index} group-mask membership is invalid")
    masks: dict[str, np.ndarray] = {}
    for name, prefixes in GROUP_PREFIXES.items():
        mask_row = mask_rows[name]
        if not isinstance(mask_row, Mapping):
            raise ValueError(f"state {index} {name} mask receipt is malformed")
        mask = _load_mask(
            _bound_output_path(
                output, mask_row.get("path"), f"state {index} {name} mask"
            ),
            mask_row.get("sha256"),
            f"state {index} {name} mask",
        )
        ids = [
            int(instance_id)
            for instance_id, path in mapping.items()
            if any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)
        ]
        expected = np.isin(instance, ids).astype(np.uint8) * 255
        if not np.array_equal(mask, expected) or int(np.count_nonzero(mask)) != mask_row.get(
            "pixel_count"
        ):
            raise ValueError(f"state {index} {name} mask differs from instance evidence")
        masks[name] = mask
    attestation = row.get("attestation")
    if not isinstance(attestation, Mapping) or attestation.get("state_index") != index:
        raise ValueError(f"state {index} runtime attestation is malformed")
    partition = attestation.get("runtime_partition")
    if not isinstance(partition, Mapping):
        raise ValueError(f"state {index} runtime partition is missing")
    validate_runtime_partition(
        partition, partition.get("prim_paths", []), partition.get("property_paths", [])
    )
    return {
        "state_index": index,
        "rgb": rgb,
        "instance": instance,
        "mapping": mapping,
        "masks": masks,
        "attestation": dict(attestation),
    }


def _validate_attestation(
    value: object,
    *,
    state_index: int,
    partition: Mapping[str, Any],
    camera_before_sha256: object,
    camera_sha256: object,
    scene_sha256: object,
) -> None:
    required = {
        "state_index",
        "source_time_code",
        "physics_steps",
        "physics_callbacks",
        "timeline_playing",
        "runtime_partition",
        "camera_light_before_annotators_sha256",
        "camera_light_sha256",
        "scene_session_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"state {state_index} runtime attestation schema is malformed")
    if (
        value.get("state_index") != state_index
        or value.get("source_time_code") != 1
        or value.get("physics_steps") != 0
        or value.get("physics_callbacks") != 0
        or value.get("timeline_playing") is not False
        or value.get("runtime_partition") != partition
        or value.get("camera_light_before_annotators_sha256") != camera_before_sha256
        or value.get("camera_light_sha256") != camera_sha256
        or value.get("scene_session_sha256") != scene_sha256
        or camera_before_sha256 != camera_sha256
    ):
        raise ValueError(f"state {state_index} runtime attestation differs from frozen evidence")


def _accepted_endpoint_evidence(
    build: Mapping[str, Any], mode: str
) -> tuple[np.ndarray | None, list[np.ndarray] | None, np.ndarray | None, list[np.ndarray] | None]:
    endpoints = build.get("adjacent_endpoints")
    if not isinstance(endpoints, Mapping):
        raise ValueError("transport build lacks adjacent endpoint bindings")
    start = endpoints.get("accepted_start")
    end = endpoints.get("accepted_end")
    start_rgb = None
    start_video = None
    end_rgb = None
    end_video = None
    if mode in {BETWEEN_MODE, EXIT_MODE}:
        if not isinstance(start, Mapping) or start.get("frame_index") != -1:
            raise ValueError("transport start endpoint is missing")
        _load_rgb(
            Path(str(start.get("path"))), start.get("sha256"), "accepted start endpoint"
        )
        start_video = _decode_video(
            Path(str(start.get("video_path"))),
            start.get("video_sha256"),
            "accepted outgoing fill video",
        )
        start_rgb = start_video[-1]
    elif start is not None:
        raise ValueError("entry transport must not bind a start endpoint")
    if mode in {ENTRY_MODE, BETWEEN_MODE}:
        if not isinstance(end, Mapping) or end.get("frame_index") != 0:
            raise ValueError("transport end endpoint is missing or not the first pre-pour frame")
        end_video = _decode_video(
            Path(str(end.get("video_path"))), end.get("video_sha256"), "accepted next fill video"
        )
        end_rgb = end_video[0]
    elif end is not None:
        raise ValueError("exit transport must not bind an end endpoint")
    return start_rgb, start_video, end_rgb, end_video


def compute_metrics(
    *, build: Mapping[str, Any], captures: Sequence[Mapping[str, Any]], full: bool
) -> dict[str, Any]:
    mode = str(build.get("mode"))
    by_index = {int(row["state_index"]): row for row in captures}
    baseline = by_index[min(by_index)]["rgb"]
    sentinel_delta = max(
        int(
            np.max(
                np.abs(
                    row["rgb"][y, x].astype(np.int16)
                    - baseline[y, x].astype(np.int16)
                )
            )
        )
        for row in captures
        for x, y in BACKGROUND_SENTINELS
    )
    result: dict[str, Any] = {
        "background_sentinel_max_channel_delta": sentinel_delta,
        "background_sentinels_within_2": sentinel_delta <= 2,
        "all_instance_mappings_and_masks_valid": True,
    }
    occupancy = _mode_occupancy_evidence(mode, captures)
    result["mode_occupancy"] = occupancy
    result["mode_occupancy_valid"] = occupancy["mode_occupancy_valid"]
    result["outgoing_absent_final"] = (
        occupancy["entry_outgoing_absent_all"]
        if mode == ENTRY_MODE
        else occupancy["outgoing_absent_late"]
    )
    if mode in {ENTRY_MODE, BETWEEN_MODE}:
        final_key = str(max(by_index))
        center = occupancy["incoming_center_x_px_by_state"][final_key]
        result["incoming_final_center_x_px"] = center
        result["incoming_center_error_px"] = (
            None if center is None else abs(center - FROZEN_FILL_ANCHOR_X_PX)
        )
        result["incoming_centered_final"] = occupancy["incoming_centered_late"]
    else:
        result["incoming_final_center_x_px"] = None
        result["incoming_center_error_px"] = None
        result["incoming_centered_final"] = None
    if not full:
        result.update(
            {
                "start_seam_mad": None,
                "start_seam_threshold_p99": None,
                "start_seam_passed": None,
                "end_seam_mad": None,
                "end_seam_threshold_p99": None,
                "end_seam_passed": None,
                "packaged_sequence_byte_unique": None,
            }
        )
        return result

    start_rgb, start_video, end_rgb, end_video = _accepted_endpoint_evidence(build, mode)
    if start_rgb is not None and start_video is not None:
        start_mad = _mad(start_rgb, by_index[1]["rgb"])
        start_threshold = _matched_p99(start_video)
        result.update(
            {
                "start_seam_mad": start_mad,
                "start_seam_threshold_p99": start_threshold,
                "start_seam_passed": start_mad <= start_threshold,
            }
        )
    else:
        result.update(
            {
                "start_seam_mad": None,
                "start_seam_threshold_p99": None,
                "start_seam_passed": True,
            }
        )
    if end_rgb is not None and end_video is not None:
        end_mad = _mad(by_index[11]["rgb"], end_rgb)
        end_threshold = _matched_p99(end_video)
        result.update(
            {
                "end_seam_mad": end_mad,
                "end_seam_threshold_p99": end_threshold,
                "end_seam_passed": end_mad <= end_threshold,
            }
        )
    else:
        result.update(
            {
                "end_seam_mad": None,
                "end_seam_threshold_p99": None,
                "end_seam_passed": True,
            }
        )
    substitution = package_state_indices(mode)
    frames: list[np.ndarray] = []
    if substitution["accepted_start_endpoint"] is not None:
        assert start_rgb is not None
        frames.append(start_rgb)
    frames.extend(by_index[index]["rgb"] for index in substitution["rendered_state_indices"])
    if substitution["accepted_end_endpoint"] is not None:
        assert end_rgb is not None
        frames.append(end_rgb)
    hashes = [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() for frame in frames]
    result["packaged_frame_count"] = len(frames)
    result["packaged_sequence_byte_unique"] = len(frames) == 13 and len(set(hashes)) == 13
    return result


def check_transport_output(
    *,
    output_dir: Path,
    segment_dir: Path,
    expected_state_indices: Sequence[int],
    evaluate_only: bool = False,
) -> dict[str, Any]:
    del evaluate_only  # The checker is deliberately read-only in both phases.
    output = output_dir.resolve()
    segment = segment_dir.resolve()
    indices = tuple(expected_state_indices)
    if indices not in (ALL_STATE_INDICES, PILOT_STATE_INDICES):
        raise ValueError("checker state set must be full 0..12 or pilot 0,6,12")
    receipt_path = output / "transport-render-receipt.json"
    receipt = _strict_object(receipt_path, "transport render receipt")
    build_path = segment / "transport-build-receipt.json"
    build = _strict_object(build_path, "transport build receipt")
    if (
        build.get("schema_version") != 1
        or build.get("artifact_type") != "fluidmix_transport_stage_build_receipt_v1"
        or build.get("status") != "pass"
        or build.get("source_time_code") != 1
        or build.get("physics_steps") != 0
        or build.get("camera_and_light_specs_authored") != []
        or build.get("mode") not in {ENTRY_MODE, BETWEEN_MODE, EXIT_MODE}
    ):
        raise ValueError("transport build receipt identity or boundary is invalid")
    manifest = build.get("layer_manifest")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(build.get("owner_thread_id"), str)
        or manifest.get("owner_thread_id") != build.get("owner_thread_id")
    ):
        raise ValueError("transport build owner tag is missing or inconsistent")
    implementation = build.get("transport_implementation")
    if (
        not isinstance(implementation, Mapping)
        or dict(implementation) != capture_transport_implementation(ROOT)
        or manifest.get("transport_implementation_sha256")
        != implementation.get("sha256")
    ):
        raise ValueError("transport implementation bytes differ from the build")
    pre_output_closure = _validate_evidence_closure_shape(
        build.get("pre_output_evidence_closure")
    )
    full = indices == ALL_STATE_INDICES
    expected_status = "candidate" if full else "pilot"
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != "fluidmix_transport_stage_render_receipt_v1"
        or receipt.get("status") != expected_status
        or receipt.get("mode") != build.get("mode")
        or receipt.get("owner_thread_id") != build.get("owner_thread_id")
        or receipt.get("state_indices") != list(indices)
        or receipt.get("source_time_code") != 1
        or receipt.get("physics_steps") != 0
        or receipt.get("physics_callbacks") != 0
        or receipt.get("build_receipt_sha256") != _sha256(build_path)
        or receipt.get("rendered_states_byte_unique") is not True
        or not _valid_sha256(receipt.get("camera_light_before_annotators_sha256"))
        or not _valid_sha256(receipt.get("camera_light_sha256"))
        or not _valid_sha256(receipt.get("scene_session_sha256"))
    ):
        raise ValueError("transport render receipt hard evidence is invalid")
    _validate_render_mode_transitions(
        receipt.get("render_mode_transitions"), expected_state_indices=indices
    )
    post_checker_closure = _validate_evidence_closure_shape(
        receipt.get("post_checker_evidence_closure")
    )
    if (
        receipt.get("pre_output_evidence_closure_sha256")
        != pre_output_closure["sha256"]
        or post_checker_closure != pre_output_closure
    ):
        raise ValueError("post-checker transport evidence differs from pre-output closure")
    readiness = receipt.get("wp1_fill_readiness")
    if not isinstance(readiness, Mapping):
        raise ValueError("transport render receipt lacks WP1 fill readiness")
    readiness_path = readiness.get("receipt_path")
    readiness_sha256 = readiness.get("receipt_sha256")
    if (
        not isinstance(readiness_path, str)
        or validate_wp1_fill_readiness(
            Path(readiness_path),
            declared_sha256=str(readiness_sha256),
            owner_thread_id=str(build["owner_thread_id"]),
        )
        != dict(readiness)
    ):
        raise ValueError("transport WP1 fill readiness binding differs")
    visual_review = receipt.get("pilot_visual_review")
    if full:
        if not isinstance(visual_review, Mapping):
            raise ValueError("full transport lacks an accepted pilot visual review")
        review_path = visual_review.get("review_path")
        if not isinstance(review_path, str) or not review_path:
            raise ValueError("full transport pilot visual review path is invalid")
        validated_review = validate_pilot_visual_review(
            Path(review_path),
            transport_implementation_sha256=str(implementation["sha256"]),
            owner_thread_id=str(build["owner_thread_id"]),
        )
        if dict(visual_review) != validated_review:
            raise ValueError("full transport pilot visual review binding differs")
    elif visual_review is not None:
        raise ValueError("pilot transport cannot contain a pilot visual review")
    partition = receipt.get("runtime_partition")
    if not isinstance(partition, Mapping):
        raise ValueError("transport render receipt lacks the frozen runtime partition")
    validate_runtime_partition(
        partition, partition.get("prim_paths", []), partition.get("property_paths", [])
    )
    capture_rows = receipt.get("captures")
    if not isinstance(capture_rows, list) or [
        row.get("state_index") for row in capture_rows
    ] != list(indices):
        raise ValueError("transport render receipt capture membership is invalid")
    preflight_rows = receipt.get("preflight_attestations")
    if (
        not isinstance(preflight_rows, list)
        or [row.get("state_index") for row in preflight_rows if isinstance(row, Mapping)]
        != list(indices)
        or len(preflight_rows) != len(indices)
    ):
        raise ValueError("transport preflight attestation membership is invalid")
    for state_index, attestation in zip(indices, preflight_rows, strict=True):
        _validate_attestation(
            attestation,
            state_index=state_index,
            partition=partition,
            camera_before_sha256=receipt["camera_light_before_annotators_sha256"],
            camera_sha256=receipt["camera_light_sha256"],
            scene_sha256=receipt["scene_session_sha256"],
        )
    _validate_attestation(
        receipt.get("final_runtime_attestation"),
        state_index=indices[-1],
        partition=partition,
        camera_before_sha256=receipt["camera_light_before_annotators_sha256"],
        camera_sha256=receipt["camera_light_sha256"],
        scene_sha256=receipt["scene_session_sha256"],
    )
    captures = [_load_capture(output, row) for row in capture_rows]
    for row in captures:
        attestation = row["attestation"]
        _validate_attestation(
            attestation,
            state_index=int(row["state_index"]),
            partition=partition,
            camera_before_sha256=receipt["camera_light_before_annotators_sha256"],
            camera_sha256=receipt["camera_light_sha256"],
            scene_sha256=receipt["scene_session_sha256"],
        )
    contact = _bound_output_path(
        output, receipt.get("contact_sheet_path"), "transport contact sheet"
    )
    if not contact.is_file() or _sha256(contact) != receipt.get("contact_sheet_sha256"):
        raise ValueError("transport contact sheet differs from its receipt")
    metrics = compute_metrics(build=build, captures=captures, full=indices == ALL_STATE_INDICES)
    gates = {
        "evidence_closure_exact": True,
        "runtime_partition_exact": True,
        "render_mode_transitions_exact": True,
        "camera_and_lighting_unchanged": True,
        "source_time_one_and_zero_physics": True,
        "instance_mappings_and_masks_valid": True,
        "background_sentinels_within_2": metrics["background_sentinels_within_2"],
        "mode_occupancy_valid": metrics["mode_occupancy_valid"],
        "start_seam_passed": metrics["start_seam_passed"],
        "end_seam_passed": metrics["end_seam_passed"],
        "packaged_sequence_byte_unique": metrics["packaged_sequence_byte_unique"],
        "pilot_visual_review_accepted": True if full else None,
    }
    if full and any(value is not True for value in gates.values()):
        raise ValueError("full transport automatic acceptance gate failed")
    return {
        "schema_version": 1,
        "artifact_type": "fluidmix_transport_stage_checker_v1",
        "status": "pass",
        "pilot": not full,
        "state_indices": list(indices),
        "metrics": metrics,
        "automatic_acceptance_gates": gates,
        "manual_visual_review_required": not full,
        "pilot_visual_review_status": "accepted" if full else "required",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_transport_output(
            output_dir=args.output_dir,
            segment_dir=args.segment_dir,
            expected_state_indices=PILOT_STATE_INDICES if args.pilot else ALL_STATE_INDICES,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"transport check failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

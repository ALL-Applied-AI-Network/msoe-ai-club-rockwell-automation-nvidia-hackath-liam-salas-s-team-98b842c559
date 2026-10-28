#!/usr/bin/env python3
"""Independently check no-physics rigid-vial transport artifacts.

The module is deliberately Isaac-free.  Renderer semantic masks are simulation
oracle evidence, not a learned camera input, and every artifact is reloaded from
disk rather than trusted from receipt claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.rigid_vial_transport import (
    ACTIVE_VIAL_PATH,
    INCOMING_VIAL_PATH,
    MAX_TRUSTED_JSON_BYTES,
    QUEUED_VIAL_PATH,
    RigidTransportSpec,
    TRANSPORT_GROUPS,
    build_rigid_transport_plan,
    load_rigid_transport_inputs,
    strict_json_loads,
    validate_authored_transport_diff,
    validate_rigid_transport_receipt,
)

FULL_FRAMES = tuple(range(21))
PILOT_FRAMES = (0, 10, 20)
FRAME_SHAPE = (480, 640, 3)
MASK_SHAPE = (480, 640)
SENTINEL_PIXELS = (
    (8, 8),
    (8, 320),
    (8, 631),
    (240, 8),
    (240, 631),
    (471, 8),
    (471, 320),
    (471, 631),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_TRUSTED_JSON_BYTES:
            raise ValueError(f"{label} exceeds the bounded JSON size limit")
        value = strict_json_loads(raw.decode("utf-8"), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is missing or unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_rgb(path: Path) -> np.ndarray:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f"transport RGB is unreadable: {path.name}")
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    if value.shape != FRAME_SHAPE or value.dtype != np.uint8:
        raise ValueError("transport RGB must be uint8 640x480 RGB")
    return value


def _load_binary_mask(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"transport instance mask is unreadable: {path.name}") from exc
    if value.shape != MASK_SHAPE or value.dtype != np.uint8:
        raise ValueError("transport instance mask must be uint8 640x480")
    if not np.all(np.isfinite(value)) or not np.all((value == 0) | (value == 1)):
        raise ValueError("transport instance mask must be finite and binary")
    return value.astype(bool)


def _load_instance_evidence(
    output_dir: Path, frame_index: int
) -> tuple[np.ndarray, dict[int, str]]:
    path = output_dir / f"instance-{frame_index:04d}.npy"
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"transport instance raster is unreadable: {path.name}") from exc
    if (
        value.shape != MASK_SHAPE
        or np.issubdtype(value.dtype, np.bool_)
        or not np.issubdtype(value.dtype, np.integer)
        or np.any(value < 0)
    ):
        raise ValueError("transport instance raster must be a non-negative integer 640x480 array")
    raw = _read_json(
        output_dir / f"instance-mapping-{frame_index:04d}.json",
        "transport instance mapping",
    )
    mapping: dict[int, str] = {}
    for raw_id, raw_path in raw.items():
        if (
            not isinstance(raw_id, str)
            or not raw_id.isdigit()
            or str(int(raw_id)) != raw_id
            or not isinstance(raw_path, str)
            or not raw_path.startswith("/")
        ):
            raise ValueError("transport instance mapping is malformed")
        instance_id = int(raw_id)
        if instance_id in mapping:
            raise ValueError("transport instance mapping contains an ambiguous ID")
        mapping[instance_id] = raw_path
    visible = {int(item) for item in np.unique(value).tolist() if int(item) != 0}
    if not visible.issubset(mapping):
        raise ValueError("transport instance raster contains unmapped IDs")
    return value, mapping


def _resolve_group_masks(
    data: np.ndarray, mapping: Mapping[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    ids_by_group: dict[str, list[int]] = {"outgoing": [], "incoming": []}
    for instance_id, instance_path in mapping.items():
        matched_groups = []
        for group_name in ids_by_group:
            matches = [
                mover
                for mover in TRANSPORT_GROUPS[group_name]
                if instance_path == mover or instance_path.startswith(mover + "/")
            ]
            if len(matches) > 1:
                raise ValueError("transport instance path ambiguously matches multiple movers")
            if matches:
                matched_groups.append(group_name)
        if len(matched_groups) > 1:
            raise ValueError("transport instance ID is ambiguous across rigid groups")
        if matched_groups:
            ids_by_group[matched_groups[0]].append(instance_id)
    outgoing = np.isin(data, ids_by_group["outgoing"])
    incoming = np.isin(data, ids_by_group["incoming"])
    return outgoing, incoming


def _decode_video(path: Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("accepted video is not decodable")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.shape != FRAME_SHAPE or frame.dtype != np.uint8:
                raise ValueError("accepted video frame has an invalid raster contract")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("accepted video needs at least two frames for splice evidence")
    return frames


def _mad(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> float:
    if left.shape != FRAME_SHAPE or right.shape != FRAME_SHAPE:
        raise ValueError("splice inputs have the wrong shape")
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    if mask is not None:
        if mask.shape != MASK_SHAPE or mask.dtype != bool or not np.any(mask):
            raise ValueError("splice measurement region is empty or malformed")
        delta = delta[mask]
    if delta.size == 0:
        raise ValueError("splice measurement region is empty")
    result = float(np.mean(delta, dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError("splice MAD is non-finite")
    return result


def _accepted_video_path(case_output: Path, accepted: Mapping[str, Any]) -> Path:
    render = accepted.get("render")
    if not isinstance(render, Mapping) or not isinstance(render.get("video_path"), str):
        raise ValueError("accepted receipt video binding is missing")
    relative = Path(render["video_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("accepted video path is unsafe")
    path = (case_output / relative).resolve()
    if case_output != path and case_output not in path.parents:
        raise ValueError("accepted video path escapes the case output")
    if not path.is_file() or _sha256(path) != render.get("video_sha256"):
        raise ValueError("accepted video bytes differ from its receipt")
    return path


def compute_automatic_gates(
    output_dir: Path,
    *,
    frame_indices: Sequence[int],
    accepted_video_frames: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Recompute pixel/mask gates from files for full or exact pilot output."""

    indices = tuple(frame_indices)
    if indices not in (FULL_FRAMES, PILOT_FRAMES):
        raise ValueError("checker requires exactly the full or reviewed pilot frames")
    frames = {index: _load_rgb(output_dir / f"frame-{index:04d}.png") for index in indices}
    outgoing = {
        index: _load_binary_mask(output_dir / f"outgoing-mask-{index:04d}.npy")
        for index in indices
    }
    incoming = {
        index: _load_binary_mask(output_dir / f"incoming-mask-{index:04d}.npy")
        for index in indices
    }
    for index in indices:
        if np.any(outgoing[index] & incoming[index]):
            raise ValueError("outgoing and incoming instance masks are ambiguous")
        instance, mapping = _load_instance_evidence(output_dir, index)
        resolved_outgoing, resolved_incoming = _resolve_group_masks(instance, mapping)
        if not np.array_equal(outgoing[index], resolved_outgoing) or not np.array_equal(
            incoming[index], resolved_incoming
        ):
            raise ValueError("stored masks differ from independently resolved instance evidence")
    if not np.any(outgoing[0]):
        raise ValueError("outgoing instance is absent at the start frame")
    if not np.any(incoming[20]):
        raise ValueError("incoming instance is absent at the final frame")

    start_x = float(np.mean(np.nonzero(outgoing[0])[1], dtype=np.float64))
    final_x = float(np.mean(np.nonzero(incoming[20])[1], dtype=np.float64))
    if not all(math.isfinite(value) for value in (start_x, final_x)):
        raise ValueError("instance center is non-finite")
    sentinel_delta = max(
        int(np.max(np.abs(frames[index][y, x].astype(np.int16) - frames[0][y, x].astype(np.int16))))
        for index in indices
        for y, x in SENTINEL_PIXELS
    )
    if any(
        outgoing[index][y, x] or incoming[index][y, x]
        for index in indices
        for y, x in SENTINEL_PIXELS
    ):
        raise ValueError("fixed-background sentinel overlaps a moving instance")

    accepted_rows = tuple(accepted_video_frames)
    if len(accepted_rows) < 2 or any(
        not isinstance(row, np.ndarray)
        or row.shape != FRAME_SHAPE
        or row.dtype != np.uint8
        for row in accepted_rows
    ):
        raise ValueError("accepted splice frames have an invalid raster contract")
    accepted_deltas = np.asarray(
        [_mad(left, right) for left, right in zip(accepted_rows, accepted_rows[1:])],
        dtype=np.float64,
    )
    if accepted_deltas.size == 0 or not np.all(np.isfinite(accepted_deltas)):
        raise ValueError("accepted in-fill splice distribution is empty")
    accepted_p99 = float(np.percentile(accepted_deltas, 99.0))
    start_splice = _mad(accepted_rows[-1], frames[0])
    end_region = ~(outgoing[20] | incoming[20])
    end_splice = _mad(accepted_rows[-1], frames[20], end_region)
    hashes = [_sha256(output_dir / f"frame-{index:04d}.png") for index in indices]
    metrics = {
        "outgoing_final_pixel_count": int(np.count_nonzero(outgoing[20])),
        "incoming_final_center_x_px": final_x,
        "active_start_center_x_px": start_x,
        "incoming_center_error_px": abs(final_x - start_x),
        "fixed_background_sentinel_max_delta": sentinel_delta,
        "accepted_in_fill_splice_mad_p99": accepted_p99,
        "start_splice_mad": start_splice,
        "end_splice_background_mad": end_splice,
        "frame_hashes": hashes,
    }
    common: dict[str, Any] = {
        "outgoing_absent_final": metrics["outgoing_final_pixel_count"] == 0,
        "incoming_centered_final": metrics["incoming_center_error_px"] <= 3.0,
        "fixed_background_sentinel": sentinel_delta <= 2,
        "start_splice": start_splice <= accepted_p99,
    }
    if indices == FULL_FRAMES:
        common.update(
            {
                "end_splice": end_splice <= accepted_p99,
                "unique_frame_hashes": len(set(hashes)) == 21,
            }
        )
    else:
        common.update({"end_splice": "pending", "unique_frame_hashes": "pending"})
    return {"automatic_acceptance_gates": common, "metrics": metrics}


def _verify_artifact_manifest(output_dir: Path, receipt: Mapping[str, Any], indices: tuple[int, ...]) -> None:
    rows = receipt.get("frames")
    if not isinstance(rows, list) or len(rows) != len(indices):
        raise ValueError("transport receipt frame manifest does not match the mode")
    for expected, row in zip(indices, rows):
        if not isinstance(row, Mapping) or row.get("frame_index") != expected:
            raise ValueError("transport receipt frame order is wrong")
        path = output_dir / f"frame-{expected:04d}.png"
        if not path.is_file() or _sha256(path) != row.get("rgb_sha256"):
            raise ValueError("transport frame hash differs from the receipt")
    actual = {path.name for path in output_dir.glob("frame-*.png")}
    expected_names = {f"frame-{index:04d}.png" for index in indices}
    if actual != expected_names:
        raise ValueError("transport output has missing or extra frames")
    for prefix, suffix in (
        ("outgoing-mask-", ".npy"),
        ("incoming-mask-", ".npy"),
        ("instance-", ".npy"),
        ("instance-mapping-", ".json"),
    ):
        actual_names = {path.name for path in output_dir.glob(f"{prefix}*{suffix}")}
        expected_names = {f"{prefix}{index:04d}{suffix}" for index in indices}
        if actual_names != expected_names:
            raise ValueError(f"transport output has missing or extra {prefix} artifacts")


def _strict_plan_and_spec(
    receipt: Mapping[str, Any], indices: tuple[int, ...]
) -> tuple[RigidTransportSpec, tuple[dict[str, object], ...]]:
    raw_spec = receipt.get("spec")
    if not isinstance(raw_spec, Mapping) or set(raw_spec) != {
        "frame_count",
        "fps",
        "outgoing_final_delta_y_m",
        "incoming_final_delta_y_m",
    }:
        raise ValueError("transport receipt spec is malformed")
    spec = RigidTransportSpec(
        frame_count=raw_spec["frame_count"],
        fps=raw_spec["fps"],
        outgoing_final_delta_y_m=raw_spec["outgoing_final_delta_y_m"],
        incoming_final_delta_y_m=raw_spec["incoming_final_delta_y_m"],
    )
    groups = receipt.get("transport_groups")
    expected_groups = {name: list(paths) for name, paths in TRANSPORT_GROUPS.items()}
    if groups != expected_groups:
        raise ValueError("transport receipt group binding is wrong")
    plan = build_rigid_transport_plan(
        TRANSPORT_GROUPS,
        {
            ACTIVE_VIAL_PATH: 0.0,
            INCOMING_VIAL_PATH: -spec.incoming_final_delta_y_m,
            QUEUED_VIAL_PATH: -0.20,
        },
        spec,
    )
    rows = receipt.get("frames")
    if not isinstance(rows, list) or len(rows) != len(indices):
        raise ValueError("transport receipt frame manifest does not match its mode")
    for index, row in zip(indices, rows):
        if not isinstance(row, Mapping):
            raise ValueError("transport receipt frame row is malformed")
        expected = dict(plan[index])
        digest = row.get("rgb_sha256")
        if set(row) != {*expected, "rgb_sha256"} or {
            key: row[key] for key in expected
        } != expected or not isinstance(digest, str):
            raise ValueError("transport receipt frame row differs from the strict plan")
    return spec, plan


def _validate_counters(receipt: Mapping[str, Any], _accepted_final_physics_step: int) -> None:
    counters = receipt.get("counters")
    integer_names = ("physics_steps", "direct_step_calls", "physics_callbacks")
    expected = {
        f"{name}_{suffix}"
        for name in integer_names
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
    if not isinstance(counters, Mapping) or set(counters) != expected:
        raise ValueError("transport counter contract is malformed")
    for name in integer_names:
        before = counters[f"{name}_before"]
        after = counters[f"{name}_after"]
        during = counters[f"{name}_during_transport"]
        if (
            type(before) is not int
            or type(after) is not int
            or type(during) is not int
            or before < 0
            or before != after
            or during != 0
        ):
            raise ValueError("transport counter contract advanced physics")
    for clock in ("simulation_time_s", "timeline_time_s"):
        before = counters[f"{clock}_before"]
        after = counters[f"{clock}_after"]
        if (
            type(before) is not float
            or type(after) is not float
            or not math.isfinite(before)
            or before < 0.0
            or before != after
        ):
            raise ValueError("transport counter contract advanced a clock")
    if (
        counters["timeline_playing_before"] is not False
        or counters["timeline_playing_after"] is not False
        or counters["source_time_code_before"] != 1
        or counters["source_time_code_after"] != 1
    ):
        raise ValueError("transport counter contract changed frozen time")


def _validate_session_diff(
    output_dir: Path,
    receipt: Mapping[str, Any],
    indices: tuple[int, ...],
    spec: RigidTransportSpec,
    plan: tuple[dict[str, object], ...],
) -> None:
    diff_path = output_dir / "session-diff.json"
    if not diff_path.is_file() or _sha256(diff_path) != receipt.get("authored_diff_sha256"):
        raise ValueError("session diff bytes differ from the receipt")
    envelope = _read_json(diff_path, "session diff")
    rows = envelope.get("frames")
    guards = envelope.get("capture_guards")
    attestations = envelope.get("state_attestations")
    if (
        set(envelope) != {"frames", "capture_guards", "state_attestations"}
        or not isinstance(rows, list)
        or not isinstance(guards, list)
        or not isinstance(attestations, list)
        or len(rows) != len(indices)
        or len(guards) != len(indices)
        or len(attestations) != len(indices)
    ):
        raise ValueError("session diff frame manifest is malformed")
    for expected_index, row in zip(indices, rows):
        if not isinstance(row, Mapping) or row.get("frame_index") != expected_index:
            raise ValueError("session diff frame order is wrong")
        validate_authored_transport_diff(row, TRANSPORT_GROUPS, plan, spec=spec)
    for expected_index, guard, attestation in zip(indices, guards, attestations):
        guard_keys = {
            "frame_index", "callbacks_before", "callbacks_after",
            "timeline_time_before_s", "timeline_time_after_s",
            "timeline_playing_before", "timeline_playing_after",
            "physics_steps_before", "physics_steps_after",
            "simulation_time_before_s", "simulation_time_after_s",
        }
        if (
            not isinstance(guard, Mapping)
            or set(guard) != guard_keys
            or guard.get("frame_index") != expected_index
            or not isinstance(attestation, Mapping)
            or set(attestation) != {"frame_index", "state"}
            or attestation.get("frame_index") != expected_index
        ):
            raise ValueError("session diff guard order is malformed")
        # Reuse the exact persisted before/after invariants without trusting aggregates.
        pairs = (
            ("callbacks_before", "callbacks_after"),
            ("physics_steps_before", "physics_steps_after"),
            ("timeline_time_before_s", "timeline_time_after_s"),
            ("simulation_time_before_s", "simulation_time_after_s"),
            ("timeline_playing_before", "timeline_playing_after"),
        )
        if any(guard.get(before) != guard.get(after) for before, after in pairs):
            raise ValueError("session diff per-frame guard advanced state")
        if any(type(guard[name]) is not int or guard[name] < 0 for name in (
            "callbacks_before", "callbacks_after", "physics_steps_before", "physics_steps_after"
        )) or any(
            type(guard[name]) is not float or not math.isfinite(guard[name]) or guard[name] < 0.0
            for name in (
                "timeline_time_before_s", "timeline_time_after_s",
                "simulation_time_before_s", "simulation_time_after_s",
            )
        ) or guard["timeline_playing_before"] is not False or guard["timeline_playing_after"] is not False:
            raise ValueError("session diff per-frame guard is malformed")
        state = attestation["state"]
        if not isinstance(state, Mapping) or set(state) != {
            "session_prim_paths", "session_property_paths", "session_sha256",
            "non_session_layer_sha256", "source_file_sha256",
        }:
            raise ValueError("session diff state attestation is malformed")
        for name in ("session_prim_paths", "session_property_paths"):
            if not isinstance(state[name], list) or state[name] != sorted(set(state[name])):
                raise ValueError("session diff state attestation is malformed")
        digest_values = [state["session_sha256"]]
        for name in ("non_session_layer_sha256", "source_file_sha256"):
            if not isinstance(state[name], Mapping) or not state[name]:
                raise ValueError("session diff state attestation is malformed")
            digest_values.extend(state[name].values())
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in digest_values
        ):
            raise ValueError("session diff state attestation is malformed")


def _verify_contact_sheet_and_optional_video(output_dir: Path, indices: tuple[int, ...]) -> None:
    import cv2

    sheet = cv2.imread(str(output_dir / "contact-sheet.png"), cv2.IMREAD_COLOR)
    if sheet is None or sheet.dtype != np.uint8 or sheet.ndim != 3 or sheet.shape[2] != 3:
        raise ValueError("transport contact sheet is unreadable")
    if indices == FULL_FRAMES:
        video = output_dir / "transport.mp4"
        if video.exists():
            capture = cv2.VideoCapture(str(video))
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                capture.release()
            if not math.isclose(fps, 10.0, rel_tol=0.0, abs_tol=1.0e-6) or count != 21:
                raise ValueError("optional transport MP4 is not 10 fps with 21 frames")
    elif (output_dir / "transport.mp4").exists():
        raise ValueError("pilot transport output must not contain an MP4")


def check_transport_output(
    *,
    output_dir: Path,
    stage: Path,
    case_output: Path,
    expected_frame_indices: Sequence[int] | None = None,
    evaluate_only: bool = False,
) -> dict[str, Any]:
    """Rehash provenance and independently recompute all available gates."""

    output = Path(output_dir).resolve()
    stage_path = Path(stage).resolve()
    case_root = Path(case_output).resolve()
    receipt_path = output / "transport-receipt.json"
    receipt = None if evaluate_only and not receipt_path.exists() else _read_json(
        receipt_path, "transport receipt"
    )
    indices = tuple(expected_frame_indices or (
        row.get("frame_index") for row in receipt.get("frames", [])
    ))
    if indices not in (FULL_FRAMES, PILOT_FRAMES):
        raise ValueError("transport receipt does not declare a reviewed capture mode")

    accepted_path = case_root / "receipt.json"
    accepted = _read_json(accepted_path, "accepted receipt")
    accepted_frames = _decode_video(_accepted_video_path(case_root, accepted))
    evaluated = compute_automatic_gates(
        output, frame_indices=indices, accepted_video_frames=accepted_frames
    )
    if evaluate_only:
        return evaluated
    assert receipt is not None
    _verify_artifact_manifest(output, receipt, indices)

    loaded = load_rigid_transport_inputs(case_root, stage_path)
    spec, plan = _strict_plan_and_spec(receipt, indices)

    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("transport receipt bindings are missing")
    if _sha256(stage_path) != bindings.get("stage_sha256"):
        raise ValueError("source stage changed after transport capture")
    if _sha256(accepted_path) != bindings.get("accepted_receipt_sha256"):
        raise ValueError("accepted receipt changed after transport capture")
    expected_bindings = {
        "stage_sha256": loaded["stage_sha256"],
        "source_layer_hashes": dict(loaded["source_layer_hashes"]),
        "accepted_receipt_sha256": loaded["receipt_sha256"],
        "accepted_video_sha256": loaded["video_sha256"],
        "final_snapshot_sha256": loaded["snapshot_sha256"],
        "accepted_final_physics_step": loaded["final_physics_step"],
    }
    if dict(bindings) != expected_bindings:
        raise ValueError("transport receipt provenance bindings were not independently reproduced")
    _validate_counters(receipt, int(loaded["final_physics_step"]))
    _validate_session_diff(output, receipt, indices, spec, plan)

    gates = receipt.get("automatic_acceptance_gates")
    recomputed = evaluated["automatic_acceptance_gates"]
    required_gates = {
        "source_and_receipt_hashes_unchanged",
        "snapshot_binding",
        "session_diff_allowlisted",
        "zero_clock_and_physics_advancement",
        *recomputed,
    }
    if not isinstance(gates, Mapping) or set(gates) != required_gates:
        raise ValueError("transport receipt gate schema is malformed")
    for key in required_gates - set(recomputed):
        if gates[key] is not True:
            raise ValueError(f"transport receipt evidence gate {key} is not proven")
    for key, value in recomputed.items():
        if gates.get(key) != value:
            raise ValueError(f"transport receipt gate {key} was not independently reproduced")
    if indices == FULL_FRAMES and any(value is not True for value in gates.values()):
        raise ValueError("full transport output has a failed automatic gate")
    if indices == FULL_FRAMES:
        expected_inputs = {
            "stage_sha256": loaded["stage_sha256"],
            "source_layer_hashes": loaded["source_layer_hashes"],
            "receipt_sha256": loaded["receipt_sha256"],
            "video_sha256": loaded["video_sha256"],
            "snapshot_sha256": loaded["snapshot_sha256"],
            "final_physics_step": loaded["final_physics_step"],
            "authored_diff_sha256": receipt["authored_diff_sha256"],
        }
        validate_rigid_transport_receipt(
            receipt, expected_inputs, TRANSPORT_GROUPS, plan, spec=spec
        )
    if indices == PILOT_FRAMES and (
        set(receipt) != {
            "schema_version", "artifact_type", "status", "part",
            "excluded_from_model_inputs", "spec", "bindings", "transport_groups",
            "frames", "authored_diff_sha256", "counters",
            "automatic_acceptance_gates", "capture_mode",
        }
        or receipt.get("capture_mode") != "pilot"
        or receipt.get("status") != "pilot_pending"
        or gates.get("end_splice") != "pending"
        or gates.get("unique_frame_hashes") != "pending"
    ):
        raise ValueError("pilot receipt must retain unavailable gates as pending")
    _verify_contact_sheet_and_optional_video(output, indices)
    return evaluated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = check_transport_output(
        output_dir=args.output_dir, stage=args.stage, case_output=args.case_output
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stitch accepted fill clips with independently accepted rendered transport states."""

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


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.transport.transport_stage import (  # noqa: E402
    BETWEEN_MODE,
    ENTRY_MODE,
    EXIT_MODE,
    build_run_transport_layout,
    package_state_indices,
    validate_pilot_visual_review,
)


FRAME_SIZE = (640, 480)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one object")
    return value


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _bound_child(root: Path, raw: object, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        raise ValueError(f"{label} path is invalid")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its render directory") from error
    return path


def _decode_video(path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(os.fspath(path))
    if not capture.isOpened():
        raise ValueError(f"fill video is not decodable: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (480, 640, 3):
                raise ValueError(f"fill video frame is not 640x480: {path}")
            frames.append(frame)
    finally:
        capture.release()
    if not frames or abs(fps - 10.0) > 1.0e-6:
        raise ValueError(f"fill video must contain 10 fps 640x480 frames: {path}")
    return frames, fps


def _mad(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))


def _matched_p99(frames: Sequence[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    return float(
        np.percentile(
            [_mad(left, right) for left, right in zip(frames, frames[1:])], 99
        )
    )


def _validated_fill_binding(
    endpoint: object, supplied: Mapping[str, Any], label: str
) -> dict[str, str]:
    if not isinstance(endpoint, Mapping):
        raise ValueError(f"transport {label} endpoint binding is missing")
    supplied_path = Path(str(supplied.get("path"))).resolve()
    supplied_hash = supplied.get("sha256")
    endpoint_path = endpoint.get("video_path")
    endpoint_hash = endpoint.get("video_sha256")
    if (
        not supplied_path.is_file()
        or _sha256(supplied_path) != supplied_hash
        or not isinstance(endpoint_path, str)
        or endpoint_path != os.fspath(supplied_path)
        or endpoint_hash != supplied_hash
    ):
        raise ValueError(f"transport {label} endpoint differs from supplied fill video")
    return {
        "case_id": str(supplied["case_id"]),
        "path": os.fspath(supplied_path),
        "sha256": str(supplied_hash),
    }


def _validate_supplied_fill_bindings(
    endpoints: Mapping[str, Any],
    *,
    mode: str,
    expected_start_fill: Mapping[str, Any] | None,
    expected_end_fill: Mapping[str, Any] | None,
) -> dict[str, object]:
    start = endpoints.get("accepted_start")
    end = endpoints.get("accepted_end")
    if mode == ENTRY_MODE:
        if start is not None or expected_start_fill is not None or expected_end_fill is None:
            raise ValueError("entry transport supplied-fill binding is malformed")
        start_binding = None
        end_binding = _validated_fill_binding(end, expected_end_fill, "entry end")
    elif mode == BETWEEN_MODE:
        if expected_start_fill is None or expected_end_fill is None:
            raise ValueError("between transport supplied-fill binding is malformed")
        start_binding = _validated_fill_binding(
            start, expected_start_fill, "between start"
        )
        end_binding = _validated_fill_binding(end, expected_end_fill, "between end")
    elif mode == EXIT_MODE:
        if end is not None or expected_start_fill is None or expected_end_fill is not None:
            raise ValueError("exit transport supplied-fill binding is malformed")
        start_binding = _validated_fill_binding(start, expected_start_fill, "exit start")
        end_binding = None
    else:
        raise ValueError("transport supplied-fill mode is invalid")
    return {
        "accepted_start_fill": start_binding,
        "accepted_end_fill": end_binding,
    }


def _accepted_transport(
    render_dir: Path,
    *,
    expected_mode: str,
    expected_case_ids: tuple[str | None, str | None],
    expected_fill_records: tuple[Mapping[str, Any] | None, Mapping[str, Any] | None],
) -> tuple[dict[int, Any], dict[str, Any]]:
    receipt_path = render_dir / "transport-render-receipt.json"
    receipt = _read_object(receipt_path, "transport render receipt")
    checker = receipt.get("checker_evaluation")
    gates = checker.get("automatic_acceptance_gates") if isinstance(checker, Mapping) else None
    visual_review = receipt.get("pilot_visual_review")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type")
        != "fluidmix_transport_stage_render_receipt_v1"
        or receipt.get("status") != "candidate"
        or receipt.get("mode") != expected_mode
        or receipt.get("state_indices") != list(range(13))
        or receipt.get("rendered_states_byte_unique") is not True
        or not isinstance(checker, Mapping)
        or checker.get("schema_version") != 1
        or checker.get("artifact_type") != "fluidmix_transport_stage_checker_v1"
        or checker.get("status") != "pass"
        or checker.get("pilot") is not False
        or checker.get("manual_visual_review_required") is not False
        or checker.get("pilot_visual_review_status") != "accepted"
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or gates.get("pilot_visual_review_accepted") is not True
        or not isinstance(visual_review, Mapping)
        or visual_review.get("status") != "accepted"
    ):
        raise ValueError("transport render is not a full accepted candidate")
    segment_dir = Path(str(receipt.get("segment_dir"))).resolve()
    build_path = segment_dir / "transport-build-receipt.json"
    build = _read_object(build_path, "transport build receipt")
    if (
        receipt.get("build_receipt_sha256") != _sha256(build_path)
        or build.get("schema_version") != 1
        or build.get("artifact_type")
        != "fluidmix_transport_stage_build_receipt_v1"
        or build.get("status") != "pass"
        or build.get("mode") != expected_mode
        or build.get("owner_thread_id") != receipt.get("owner_thread_id")
    ):
        raise ValueError("transport render mode/build binding is invalid")
    implementation = build.get("transport_implementation")
    review_path = visual_review.get("review_path")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(review_path, str)
        or validate_pilot_visual_review(
            Path(review_path),
            transport_implementation_sha256=str(implementation.get("sha256")),
            owner_thread_id=str(build.get("owner_thread_id")),
        )
        != dict(visual_review)
    ):
        raise ValueError("transport pilot visual review binding is invalid")
    endpoints = build.get("adjacent_endpoints")
    if not isinstance(endpoints, Mapping):
        raise ValueError("transport build lacks endpoint bindings")
    outgoing_case, next_case = expected_case_ids
    for key, expected in (
        ("outgoing_case_output", outgoing_case),
        ("next_case_output", next_case),
    ):
        value = endpoints.get(key)
        if expected is None:
            if value is not None:
                raise ValueError("transport endpoint unexpectedly binds a case")
        elif (
            not isinstance(value, str)
            or Path(value).resolve().parent.name != expected
        ):
            raise ValueError("transport endpoint case identity differs from the run layout")
    fill_bindings = _validate_supplied_fill_bindings(
        endpoints,
        mode=expected_mode,
        expected_start_fill=expected_fill_records[0],
        expected_end_fill=expected_fill_records[1],
    )
    frames: dict[int, Any] = {}
    captures = receipt.get("captures")
    if (
        not isinstance(captures, list)
        or len(captures) != 13
        or [row.get("state_index") for row in captures if isinstance(row, Mapping)]
        != list(range(13))
    ):
        raise ValueError("transport receipt lacks its exact 13 ordered captures")
    for row in captures:
        if not isinstance(row, Mapping):
            raise ValueError("transport capture row is malformed")
        path = _bound_child(render_dir, row.get("rgb_path"), "transport RGB")
        if not path.is_file() or _sha256(path) != row.get("rgb_sha256"):
            raise ValueError("transport RGB differs from its receipt")
        frame = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (480, 640, 3):
            raise ValueError("transport RGB is not decodable")
        frames[int(row["state_index"])] = frame
    if set(frames) != set(range(13)):
        raise ValueError("transport render lacks one of its 13 states")
    return frames, {
        "render_dir": os.fspath(render_dir.resolve()),
        "receipt_path": os.fspath(receipt_path.resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "mode": expected_mode,
        "fill_video_bindings": fill_bindings,
    }


def compose_run_video(
    *,
    run_id: str,
    case_ids: Sequence[str],
    fill_videos: Sequence[Path],
    transport_render_dirs: Sequence[Path],
    output_mp4: Path,
) -> dict[str, Any]:
    """Compose N fills with entry, N-1 between segments, and one exit."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run composition requires a non-empty run id")
    layout = build_run_transport_layout(case_ids)
    if len(fill_videos) != len(case_ids) or len(transport_render_dirs) != len(layout):
        raise ValueError("run composition requires N fills and N+1 transport segments")
    output = output_mp4.resolve()
    receipt_path = output.with_suffix(".run.json")
    partial = output.with_name(output.stem + ".partial.mp4")
    if output.exists() or receipt_path.exists() or partial.exists():
        raise ValueError("run composition outputs must not already exist")
    fills: list[list[Any]] = []
    fill_records: list[dict[str, Any]] = []
    for case_id, raw_path in zip(case_ids, fill_videos, strict=True):
        path = raw_path.resolve()
        frames, fps = _decode_video(path)
        fills.append(frames)
        fill_records.append(
            {
                "case_id": case_id,
                "path": os.fspath(path),
                "sha256": _sha256(path),
                "frame_count": len(frames),
                "fps": fps,
            }
        )
    transports: list[dict[int, Any]] = []
    transport_records: list[dict[str, Any]] = []
    transport_fill_bindings: list[dict[str, Any]] = []
    case_positions = {case_id: index for index, case_id in enumerate(case_ids)}
    for row, raw_dir in zip(layout, transport_render_dirs, strict=True):
        outgoing_case = row["outgoing_case_id"]
        next_case = row["next_case_id"]
        frames, record = _accepted_transport(
            raw_dir.resolve(),
            expected_mode=str(row["mode"]),
            expected_case_ids=(outgoing_case, next_case),
            expected_fill_records=(
                None
                if outgoing_case is None
                else fill_records[case_positions[str(outgoing_case)]],
                None
                if next_case is None
                else fill_records[case_positions[str(next_case)]],
            ),
        )
        transports.append(frames)
        fill_binding = record.pop("fill_video_bindings")
        transport_fill_bindings.append(
            {
                "segment_index": row["segment_index"],
                "mode": row["mode"],
                **fill_binding,
            }
        )
        transport_records.append(record)

    sequence: list[Any] = []
    encoded_transition_ranges: list[dict[str, int | str]] = []
    sequence.extend(
        transports[0][index]
        for index in package_state_indices(ENTRY_MODE)["rendered_state_indices"]
    )
    sequence.extend(fills[0])
    encoded_transition_ranges.append(
        {"mode": ENTRY_MODE, "start_frame_index": 0, "stop_frame_index": 13}
    )
    for fill_index in range(len(fills) - 1):
        transition_start = len(sequence) - 1
        between = transports[fill_index + 1]
        sequence.extend(
            between[index]
            for index in package_state_indices(BETWEEN_MODE)["rendered_state_indices"]
        )
        sequence.extend(fills[fill_index + 1])
        encoded_transition_ranges.append(
            {
                "mode": BETWEEN_MODE,
                "start_frame_index": transition_start,
                "stop_frame_index": transition_start + 13,
            }
        )
    exit_start = len(sequence) - 1
    sequence.extend(
        transports[-1][index]
        for index in package_state_indices(EXIT_MODE)["rendered_state_indices"]
    )
    encoded_transition_ranges.append(
        {
            "mode": EXIT_MODE,
            "start_frame_index": exit_start,
            "stop_frame_index": exit_start + 13,
        }
    )
    expected_count = (
        sum(len(frames) for frames in fills)
        + 12
        + 11 * (len(fills) - 1)
        + 12
    )
    if len(sequence) != expected_count:
        raise RuntimeError("endpoint-substituted run sequence frame count drifted")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = cv2.VideoWriter(
            os.fspath(partial), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, FRAME_SIZE
        )
        if not writer.isOpened():
            raise RuntimeError("run MP4 writer did not open")
        try:
            for frame in sequence:
                writer.write(frame)
        finally:
            writer.release()
        partial.replace(output)
        decoded, fps = _decode_video(output)
        if len(decoded) != expected_count or fps != 10.0:
            raise RuntimeError("composed run video failed decode/count verification")
        encoded_transition_evidence: list[dict[str, Any]] = []
        for expected_segment, bounds in zip(
            layout, encoded_transition_ranges, strict=True
        ):
            start = int(bounds["start_frame_index"])
            stop = int(bounds["stop_frame_index"])
            frames = decoded[start:stop]
            hashes = [
                hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
                for frame in frames
            ]
            byte_unique = len(frames) == 13 and len(set(hashes)) == 13
            encoded_transition_evidence.append(
                {
                    "segment_index": expected_segment["segment_index"],
                    "mode": expected_segment["mode"],
                    "start_frame_index": start,
                    "stop_frame_index": stop,
                    "frame_count": len(frames),
                    "decoded_bgr_sha256": hashes,
                    "byte_unique": byte_unique,
                }
            )
        if any(row["byte_unique"] is not True for row in encoded_transition_evidence):
            raise RuntimeError("encoded run transport sequence is not byte-unique")
        encoded_seam_evidence: list[dict[str, Any]] = []
        for segment, bounds in zip(layout, encoded_transition_ranges, strict=True):
            start = int(bounds["start_frame_index"])
            stop = int(bounds["stop_frame_index"])
            frames = decoded[start:stop]
            outgoing_case = segment["outgoing_case_id"]
            next_case = segment["next_case_id"]
            start_mad = None
            start_threshold = None
            start_passed = True
            if outgoing_case is not None:
                start_mad = _mad(frames[0], frames[1])
                start_threshold = _matched_p99(fills[case_positions[str(outgoing_case)]])
                start_passed = start_mad <= start_threshold
            end_mad = None
            end_threshold = None
            end_passed = True
            if next_case is not None:
                end_mad = _mad(frames[11], frames[12])
                end_threshold = _matched_p99(fills[case_positions[str(next_case)]])
                end_passed = end_mad <= end_threshold
            encoded_seam_evidence.append(
                {
                    "segment_index": segment["segment_index"],
                    "mode": segment["mode"],
                    "start_seam_mad": start_mad,
                    "start_seam_threshold_p99": start_threshold,
                    "start_seam_passed": start_passed,
                    "end_seam_mad": end_mad,
                    "end_seam_threshold_p99": end_threshold,
                    "end_seam_passed": end_passed,
                }
            )
        if any(
            row["start_seam_passed"] is not True
            or row["end_seam_passed"] is not True
            for row in encoded_seam_evidence
        ):
            raise RuntimeError("encoded run transport seam gate failed")
        receipt = {
            "schema_version": 2,
            "artifact_type": "rendered_transport_run_sequence_v2",
            "status": "pass",
            "run_id": run_id,
            "case_ids": list(case_ids),
            "fill_clips": fill_records,
            "transport_segments": transport_records,
            "transport_fill_video_bindings": transport_fill_bindings,
            "transport_segment_count": len(layout),
            "entry_segment_count": 1,
            "between_segment_count": len(case_ids) - 1,
            "exit_segment_count": 1,
            "endpoint_substitution": {
                mode: package_state_indices(mode)
                for mode in (ENTRY_MODE, BETWEEN_MODE, EXIT_MODE)
            },
            "output_video": os.fspath(output),
            "output_video_sha256": _sha256(output),
            "output_frame_count": expected_count,
            "fps": 10.0,
            "resolution": [640, 480],
            "encoded_transition_evidence": encoded_transition_evidence,
            "encoded_transition_sequences_byte_unique": True,
            "encoded_seam_evidence": encoded_seam_evidence,
            "encoded_transition_seams_passed": True,
            "qualified_fill_receipts_changed": False,
            "camera_and_lighting_changed": False,
        }
        _write_json_new(receipt_path, receipt)
    except BaseException:
        for owned in (partial, output, receipt_path):
            if owned.is_file():
                owned.unlink()
        raise
    return receipt


def compose_case_video(**_kwargs: object) -> dict[str, Any]:
    """Fail closed if an old caller attempts the rejected 2D compositor."""

    raise ValueError(
        "the 2D post-fill compositor is retired; compose an accepted rendered run sequence"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--fill-video", type=Path, action="append", required=True)
    parser.add_argument("--transport-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compose_run_video(
            run_id=args.run_id,
            case_ids=args.case_id,
            fill_videos=args.fill_video,
            transport_render_dirs=args.transport_dir,
            output_mp4=args.output_mp4,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"run composition failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

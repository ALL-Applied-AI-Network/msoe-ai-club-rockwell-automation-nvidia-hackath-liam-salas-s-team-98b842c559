#!/usr/bin/env python3
"""Append fixed-camera vial transport to one accepted PhysX case video."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.vial_transport import (  # noqa: E402
    TransportSpec,
    compose_transport_frames,
    extract_fillcam_mover_masks,
    transport_offsets,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> dict:
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _write_json_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def compose_case_video(
    *, case_output: Path, output_mp4: Path, spec: TransportSpec | None = None
) -> dict:
    """Append transport frames to one accepted case and return its receipt."""

    case_output = case_output.resolve()
    output_mp4 = output_mp4.resolve()
    receipt_path = case_output / "receipt.json"
    receipt = _read_object(receipt_path, "case receipt")
    if receipt.get("status") != "pass":
        raise ValueError("case receipt status is not pass")
    render = receipt.get("render")
    semantics = receipt.get("semantics")
    if not isinstance(render, dict) or not isinstance(semantics, dict):
        raise ValueError("case receipt lacks render or semantic evidence")
    pairs = semantics.get("pairs")
    if not isinstance(pairs, list) or not pairs or not isinstance(pairs[-1], dict):
        raise ValueError("case receipt lacks a final RGB/mask pair")
    final_pair = pairs[-1]
    source_video = case_output / str(render.get("video_path", ""))
    final_rgb_path = case_output / str(final_pair.get("rgb_path", ""))
    liquid_mask_path = case_output / str(final_pair.get("mask_path", ""))
    for path, description in (
        (source_video, "source video"),
        (final_rgb_path, "final RGB"),
        (liquid_mask_path, "final liquid mask"),
    ):
        if not path.is_file():
            raise ValueError(f"{description} is missing: {path}")
    if _sha256(source_video) != render.get("video_sha256"):
        raise ValueError("source video hash differs from its accepted receipt")
    if _sha256(final_rgb_path) != final_pair.get("rgb_sha256"):
        raise ValueError("final RGB hash differs from its accepted receipt")
    if _sha256(liquid_mask_path) != final_pair.get("mask_sha256"):
        raise ValueError("final liquid mask hash differs from its accepted receipt")

    frame_bgr = cv2.imread(str(final_rgb_path), cv2.IMREAD_COLOR)
    liquid = cv2.imread(str(liquid_mask_path), cv2.IMREAD_GRAYSCALE)
    if frame_bgr is None or liquid is None:
        raise ValueError("final RGB or liquid mask is not decodable")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    outgoing, incoming = extract_fillcam_mover_masks(frame_rgb, liquid)

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise ValueError("source video is not decodable")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        capture.release()
        raise ValueError("source video FPS is invalid")
    active_spec = spec or TransportSpec(fps=source_fps)
    if not np.isclose(active_spec.fps, source_fps, rtol=0.0, atol=1.0e-6):
        capture.release()
        raise ValueError("transport FPS differs from source video FPS")
    transport_frames = compose_transport_frames(
        frame_rgb,
        outgoing_mask=outgoing,
        incoming_mask=incoming,
        spec=active_spec,
    )

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    transport_receipt_path = output_mp4.with_suffix(".transport.json")
    partial = output_mp4.with_name(f"{output_mp4.stem}.partial.mp4")
    if output_mp4.exists() or transport_receipt_path.exists() or partial.exists():
        capture.release()
        raise ValueError("transport output paths must not already exist")
    writer = cv2.VideoWriter(
        str(partial),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (640, 480),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("MP4 writer did not open")
    source_frame_count = 0
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok:
                break
            if source_frame.shape != (480, 640, 3):
                raise ValueError("source video frame is not 640x480 RGB")
            writer.write(source_frame)
            source_frame_count += 1
        if source_frame_count == 0:
            raise ValueError("source video contains no frames")
        for transport_rgb in transport_frames[1:]:
            writer.write(cv2.cvtColor(transport_rgb, cv2.COLOR_RGB2BGR))
    finally:
        capture.release()
        writer.release()
    partial.replace(output_mp4)

    transport_hashes = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in transport_frames
    ]
    result = {
        "schema_version": 1,
        "artifact_type": "run_sequence_vial_transport_v1",
        "status": "pass",
        "source_case_receipt": str(receipt_path),
        "source_case_receipt_sha256": _sha256(receipt_path),
        "source_video": str(source_video),
        "source_video_sha256": _sha256(source_video),
        "source_frame_count": source_frame_count,
        "output_video": str(output_mp4),
        "output_video_sha256": _sha256(output_mp4),
        "output_frame_count": source_frame_count + len(transport_frames) - 1,
        "fps": source_fps,
        "transport_frame_count": len(transport_frames),
        "transport_frame_rgb_sha256": transport_hashes,
        "transport_frames_byte_unique": len(set(transport_hashes))
        == len(transport_hashes),
        "transport_offsets_px": [list(row) for row in transport_offsets(active_spec)],
        "outgoing_mask_pixel_count": int(np.count_nonzero(outgoing)),
        "incoming_mask_pixel_count": int(np.count_nonzero(incoming)),
        "camera_transform_changed": False,
        "qualified_fill_receipt_changed": False,
        "scope": "post_fill_rigid_vial_carrier_transport_only",
    }
    if not result["transport_frames_byte_unique"]:
        raise RuntimeError("transport frame RGB bytes are not unique")
    _write_json_new(transport_receipt_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--transport-frames", type=int, default=21)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compose_case_video(
            case_output=args.case_output,
            output_mp4=args.output_mp4,
            spec=TransportSpec(frame_count=args.transport_frames),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"transport failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

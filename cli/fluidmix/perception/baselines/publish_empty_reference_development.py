#!/usr/bin/env python3
"""Publish frozen Approach-A predictions before any semantic mask is opened."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.fspath(ROOT / "src"))

_SAFE_MANIFEST = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "rgb-mask-predictor-01a18128/approach-a-stage2-partition-fix-attempt-01a18128/"
    "development-inputs-v1.json"
)
_SAFE_MANIFEST_SHA256 = "a39f5ec1e3dc659f894cec7a319d9dd1ad0c9ddb5635d5b8dd46c499e74a46c6"
_PREDICTOR_SHA256 = "1e933c6eb8e6a729c820211c190aaa72f8d5ea203706e503484e3531800d80a1"
_SOURCE_ROOT = Path("/home/hackathon5/liamreporoot/f2-varied-cohort-merged-9e6254d")
_AUTHORIZED = {
    "CASE_afc8ed07715ca816": {
        "vial_id": "V1",
        "video_sha256": "643f5ed00f3ed608b3cd1c2648e23cd2b59865cd1db3b72b0d72583f1d44d6f7",
        "receipt_sha256": "dc8576bed42d84b68658a356b79ea86569bda8a91b982a313a4a22dc0e1bc6b4",
        "prefill_png_sha256": "4609928fa9f70ff06882bed9753671ff98b0d26bedab3610099ef7a4fec86af4",
        "prefill_rgb_sha256": "b88e7b90f6194aba6b6ac171bb5493fdaf745dc7762d0b6567feadb002819897",
        "stages": (
            (
                "frames/frame-0001-step-03136-t200-stage.png",
                "8e5c7bb96e494e6c27b81861d1bbe3de393a85e94cb10351f111692cb18efc0f",
            ),
            (
                "frames/frame-0002-step-04612-t400-stage.png",
                "13493ac87f723c5d6461e9b0052a5a99991135a4c833cd9a485202a9d4324d19",
            ),
        ),
    },
    "CASE_d170a37f08ef48ed": {
        "vial_id": "V2",
        "video_sha256": "a1f478e2d2057073303fe20b094f197c84b5ca1e6fe7c30e11b9f46a75e5d9ae",
        "receipt_sha256": "0dde133e01cb38938db8d98645ace153ba05c88a2a02016bcbbcafc3f3f4417d",
        "prefill_png_sha256": "6c26a857f28a926ea096f3ebad7aa51a4d485a24912afa91894c5a8dde949d78",
        "prefill_rgb_sha256": "547e1b15a39634d9b8fcce7c849284749a68d869423c1b10f8025769663d4431",
        "stages": (
            (
                "frames/frame-0001-step-03136-t200-stage.png",
                "c3b708fe0fb23d5d9b40391d89ac49d72441d818feb4b9dcb0ec3b9f9fc9b4ff",
            ),
            (
                "frames/frame-0002-step-04612-t400-stage.png",
                "142b1f26bfa9083e5b12ee4f5771afab43cbd54cc915f4ccbbeb12c85b56748d",
            ),
        ),
    },
    "CASE_82033c6cae4d93da": {
        "vial_id": "V3",
        "video_sha256": "98c35cabaaf35c2720bace039c6243e4dd40b065a8b55965518c3d6d691c4477",
        "receipt_sha256": "7e125459647729cc68ad7d4b0ea69cc8a24b4e892fe992065e04985b78bc2948",
        "prefill_png_sha256": "364f13a5b635683a3681430d90dc6c19465784e881f3456256398dea22686d97",
        "prefill_rgb_sha256": "9f8328be8520d203cea7f191690eb906bb4bae79734bc3f787d2a5f19295ee4a",
        "stages": (
            (
                "frames/frame-0001-step-03136-t200-stage.png",
                "804e479fef485078d5911736d772144ad6927d72c20b4ea295304d29e8544d76",
            ),
            (
                "frames/frame-0002-step-04612-t400-stage.png",
                "ed36965c8bf0e263cc6b2af80b32205ea0b5f3a86c34d492ffbc75e1bffe174f",
            ),
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_png(path: Path, digest: str) -> np.ndarray:
    if path.is_symlink() or _sha256(path) != digest:
        raise ValueError(f"PNG hash mismatch: {path}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
            raise ValueError(f"expected exact 640x480 RGB PNG: {path}")
        image.load()
        return np.asarray(image, dtype=np.uint8).copy()


def _decode_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(os.fspath(path))
    frames = []
    try:
        for _ in range(count):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"video has fewer than {count} frames: {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
                raise ValueError(f"decoded frame violates FillCam RGB contract: {path}")
            frames.append(rgb)
    finally:
        capture.release()
    return frames


def _mask_record(mask: np.ndarray) -> dict[str, object]:
    packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    return {
        "shape_hw": [480, 640],
        "packbits_bitorder": "little",
        "packed_sha256": hashlib.sha256(packed).hexdigest(),
        "packed_base64": base64.b64encode(packed).decode("ascii"),
        "pixel_count": int(np.count_nonzero(mask)),
    }


def _publish_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _same_outputs(left, right) -> bool:
    return all(
        a.status == b.status
        and a.surface_row == b.surface_row
        and a.visible_height_px == b.visible_height_px
        and a.median_rgb == b.median_rgb
        and np.array_equal(a.mask, b.mask)
        for a, b in zip(left, right, strict=True)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.safe_manifest.resolve() != _SAFE_MANIFEST or _sha256(args.safe_manifest) != _SAFE_MANIFEST_SHA256:
        raise ValueError("safe manifest is not the frozen validation-only development artifact")
    predictor_path = ROOT / "src/fluidmix/perception/baselines/empty_reference_rgb_predictor.py"
    if _sha256(predictor_path) != _PREDICTOR_SHA256:
        raise ValueError("predictor source bytes differ from the frozen development implementation")

    from fluidmix.perception.baselines.empty_reference_rgb_predictor import (
        CHANGE_THRESHOLD,
        CLAMP_ROWS,
        fillcam_masks,
        predict_fill_sequence,
        predict_visible_liquid,
    )

    manifest = json.loads(args.safe_manifest.read_bytes())
    development = manifest.get("records")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "empty_reference_rgb_development_inputs"
        or manifest.get("partition") != "validation"
        or not isinstance(development, list)
        or len(development) != 3
        or any(not isinstance(row, dict) for row in development)
        or any(row.get("split_role") != "validation" for row in development)
        or {row.get("case_id") for row in development} != set(_AUTHORIZED)
        or sum(len(row.get("stages", [])) for row in development) != 6
    ):
        raise ValueError("expected the exact validation-only development input artifact")

    roi, clamp = fillcam_masks()
    records = []
    gate_values = []
    for record in development:
        authorized = _AUTHORIZED[record["case_id"]]
        case_root = _SOURCE_ROOT / "cases" / record["case_id"] / "app-output"
        video_path = case_root / "three-pour.avi"
        expected_identity = {
            "batch_id": "BATCH_81a5e5e7197a8fe5",
            "run_id": "RUN-VALIDATION-01",
            "vial_id": authorized["vial_id"],
            "split_role": "validation",
            "video_path": os.fspath(video_path),
            "video_sha256": authorized["video_sha256"],
            "source_receipt_sha256": authorized["receipt_sha256"],
            "fill_frame_range": [0, 381],
        }
        if any(record.get(key) != value for key, value in expected_identity.items()):
            raise ValueError(f"development record identity mismatch: {record['case_id']}")
        receipt_path = case_root / "receipt.json"
        if receipt_path.is_symlink() or _sha256(receipt_path) != authorized["receipt_sha256"]:
            raise ValueError(f"receipt hash mismatch: {record['case_id']}")
        receipt = json.loads(receipt_path.read_bytes())
        parity = receipt.get("pre_pour_parity", {})
        source_steps = receipt.get("render", {}).get("video_timing_source_frame_steps", [])
        capture_indices = receipt.get("render", {}).get("video_frame_source_capture_indices", [])
        if (
            parity.get("passed") is not True
            or parity.get("f2_prefill_path") != "pre-pour-parity/f2-prefill-time-0001.png"
            or parity.get("f2_prefill_sha256") != authorized["prefill_png_sha256"]
            or not source_steps
            or source_steps[0] != 12
            or not capture_indices
            or capture_indices[0] != 0
            or receipt.get("render", {}).get("resolution") != [640, 480]
            or receipt.get("timing", {}).get("physics_hz") != 120
            or receipt.get("render", {}).get("video_playback_fps") != 10.0
        ):
            raise ValueError(f"pre-pour/video provenance mismatch: {record['case_id']}")
        if source_steps[0] <= 0:
            raise ValueError("decoded video frame 0 must not be accepted as the pre-pour reference")

        reference_path = case_root / parity["f2_prefill_path"]
        reference = _load_png(reference_path, authorized["prefill_png_sha256"])
        reference_sha = hashlib.sha256(reference.tobytes()).hexdigest()
        if reference_sha != authorized["prefill_rgb_sha256"]:
            raise ValueError(f"prefill RGB-array hash mismatch: {record['case_id']}")
        empty_control_path = case_root / parity.get("reference_path", "")
        empty_control = _load_png(empty_control_path, parity.get("reference_sha256", ""))
        if empty_control_path == reference_path:
            raise ValueError("empty control must be a different same-run capture path")

        if _sha256(video_path) != authorized["video_sha256"]:
            raise ValueError(f"development video hash mismatch: {record['case_id']}")
        frames = _decode_frames(video_path, record["fill_frame_range"][1])
        frame_zero_sha = hashlib.sha256(frames[0].tobytes()).hexdigest()
        if frame_zero_sha == reference_sha:
            raise ValueError("step-12 video frame unexpectedly aliases the pre-pour reference")

        low_sequence = predict_fill_sequence(
            [reference, *frames],
            expected_reference_rgb_sha256=reference_sha,
            programmed_ml=10.0,
        )
        high_sequence = predict_fill_sequence(
            [reference, *frames],
            expected_reference_rgb_sha256=reference_sha,
            programmed_ml=70.0,
        )
        sequence_invariant = _same_outputs(low_sequence, high_sequence)
        empty_result = predict_visible_liquid(empty_control, reference, programmed_ml=50.0)
        black_result = predict_visible_liquid(np.zeros_like(reference), reference, programmed_ml=50.0)
        raw_empty_difference = np.max(
            np.abs(empty_control.astype(np.int16) - reference.astype(np.int16)), axis=2
        )

        stages = []
        for stage, (relative_rgb, rgb_sha) in zip(record["stages"], authorized["stages"], strict=True):
            rgb_path = case_root / relative_rgb
            rgb = _load_png(rgb_path, rgb_sha)
            programmed_total = float(sum(record["recipe_ml"].values()))
            prediction = predict_visible_liquid(rgb, reference, programmed_ml=programmed_total)
            low = predict_visible_liquid(rgb, reference, programmed_ml=10.0)
            high = predict_visible_liquid(rgb, reference, programmed_ml=70.0)
            stages.append(
                {
                    "image_id": stage["image_id"],
                    "stage_ordinal": stage["stage_ordinal"],
                    "tank_id": stage["tank_id"],
                    "rgb_path": os.fspath(rgb_path),
                    "rgb_sha256": rgb_sha,
                    "status": prediction.status,
                    "surface_row": prediction.surface_row,
                    "visible_height_px": prediction.visible_height_px,
                    "median_rgb": prediction.median_rgb,
                    "changed_pixel_count": prediction.changed_pixel_count,
                    "above_clamp_changed_pixels": int(np.count_nonzero(prediction.mask[: CLAMP_ROWS[0]])),
                    "below_clamp_changed_pixels": int(np.count_nonzero(prediction.mask[CLAMP_ROWS[1] + 1 :])),
                    "programmed_ml_output_invariant": (
                        low.status == high.status
                        and low.surface_row == high.surface_row
                        and low.visible_height_px == high.visible_height_px
                        and low.median_rgb == high.median_rgb
                        and np.array_equal(low.mask, high.mask)
                    ),
                    "prediction_mask": _mask_record(prediction.mask),
                }
            )

        checks = {
            "video_frame_zero_rejected_as_reference": source_steps[0] == 12 and frame_zero_sha != reference_sha,
            "empty_different_capture_abstains": empty_result.surface_row is None,
            "black_frame_abstains": black_result.status == "invalid_black_frame",
            "programmed_ml_stage_outputs_invariant": all(row["programmed_ml_output_invariant"] for row in stages),
            "programmed_ml_full_sequence_outputs_invariant": sequence_invariant,
            "stage_one_lower_only": (
                stages[0]["status"] == "visible_measured"
                and stages[0]["above_clamp_changed_pixels"] == 0
                and stages[0]["below_clamp_changed_pixels"] > 0
            ),
            "stage_two_two_regions": (
                stages[1]["status"] == "visible_measured"
                and stages[1]["above_clamp_changed_pixels"] > 0
                and stages[1]["below_clamp_changed_pixels"] > 0
            ),
            "no_clamp_pixels": all(
                not np.any(
                    np.unpackbits(
                        np.frombuffer(base64.b64decode(row["prediction_mask"]["packed_base64"]), dtype=np.uint8),
                        bitorder="little",
                    )[: 480 * 640].reshape(480, 640).astype(bool)
                    & clamp
                )
                for row in stages
            ),
        }
        gate_values.extend(checks.values())
        records.append(
            {
                "case_id": record["case_id"],
                "batch_id": record["batch_id"],
                "run_id": record["run_id"],
                "vial_id": record["vial_id"],
                "recipe_ml": record["recipe_ml"],
                "source_receipt_path": os.fspath(receipt_path),
                "source_receipt_sha256": authorized["receipt_sha256"],
                "reference_source": "pre_pour_parity.f2_prefill_path",
                "reference_path": os.fspath(reference_path),
                "reference_png_sha256": authorized["prefill_png_sha256"],
                "reference_rgb_array_sha256": reference_sha,
                "empty_control_path": os.fspath(empty_control_path),
                "empty_control_png_sha256": parity["reference_sha256"],
                "empty_control_raw_changed_roi_pixels": int(
                    np.count_nonzero((raw_empty_difference > CHANGE_THRESHOLD) & roi)
                ),
                "video_path": os.fspath(video_path),
                "video_sha256": authorized["video_sha256"],
                "video_frame_zero_physics_step": source_steps[0],
                "video_frame_zero_rgb_sha256": frame_zero_sha,
                "checks": checks,
                "sequence": {
                    "reference_frame_count": 1,
                    "decoded_video_frame_count": len(frames),
                    "processed_frame_count": len(low_sequence),
                    "hidden_inferred_frame_count": sum(row.status == "clamp_hidden_inferred" for row in low_sequence),
                    "status_counts": {
                        status: sum(row.status == status for row in low_sequence)
                        for status in sorted({row.status for row in low_sequence})
                    },
                },
                "stages": stages,
            }
        )

    artifact = {
        "schema_version": 2,
        "artifact_type": "empty_reference_rgb_frozen_predictions",
        "status": "prediction_gates_pass" if all(gate_values) else "prediction_gates_fail",
        "approach": "A_same_run_prepour_empty_reference_difference",
        "bindings": {
            "safe_manifest_path": os.fspath(_SAFE_MANIFEST),
            "safe_manifest_sha256": _SAFE_MANIFEST_SHA256,
            "predictor_source_sha256": _PREDICTOR_SHA256,
            "publisher_source_sha256": _sha256(Path(__file__)),
        },
        "phase_separation": {
            "phase": "rgb_only_prediction_publish",
            "semantic_oracle_paths_read": 0,
            "semantic_oracle_images_opened": 0,
            "scoring_requires_this_artifact_sha256": True,
        },
        "denominators": {
            "development_videos": len(records),
            "development_stage_rgb_images": sum(len(row["stages"]) for row in records),
            "held_out_images_opened": 0,
            "held_out_videos_opened": 0,
            "video5_inputs_opened": 0,
        },
        "development_records": records,
        "boundary": (
            "RGB-only development predictions are frozen before the separate scorer may open "
            "validation semantic labels; heldout evaluation and video 5 remain sealed."
        ),
    }
    _publish_exclusive(args.out, artifact)
    print(f"{artifact['status']}: videos=3 stages=6; output={args.out}; sha256={_sha256(args.out)}")
    return 0 if artifact["status"] == "prediction_gates_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

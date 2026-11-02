#!/usr/bin/env python3
"""Score the frozen FillCam model/rule/hand ruler on 27 real-check windows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fluidmix import fill_level_real_eval as real_eval  # noqa: E402
from fluidmix import fill_level_surface_eval as surface_eval  # noqa: E402
from fluidmix import fill_level_test_eval as test_eval  # noqa: E402
from fluidmix import fill_level_training as training  # noqa: E402


EXPECTED_VIDEO_SHA256 = (
    "6f87cec1aeec3b2bae6dac639a317c077a3f175f27abd5fc8430d6953d9d05b9"
)
EXPECTED_REAL_VIALS = 27
EXPECTED_REAL_FRAMES = 1470
EXPECTED_WIDTH = 640
EXPECTED_HEIGHT = 480
LAST_REQUIRED_FRAME = 50874


def _append_state(path, status, **details):
    value = {
        "status": status,
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **details,
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _json_dump(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def _load_model(torch, checkpoint_path, binding, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported training checkpoint")
    if int(checkpoint.get("epoch", -1)) != binding["best_epoch"]:
        raise ValueError("checkpoint epoch does not match training summary")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("training checkpoint has no config")
    if (
        config.get("crop_xyxy") != binding["crop_xyxy"]
        or config.get("class_ids") != binding["class_ids"]
        or int(config.get("base_channels", -1)) != binding["base_channels"]
    ):
        raise ValueError("checkpoint model contract does not match the frozen binding")
    model = training.build_unet(base_channels=binding["base_channels"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--windows-sha256", required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--recipes-sha256", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-sha256", required=True)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--batch-summary-sha256", required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-summary-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--training-module-sha256", required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--frozen-rule-sha256", required=True)
    parser.add_argument("--frozen-ruler", type=Path, required=True)
    parser.add_argument("--frozen-ruler-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def _window_value(window, name):
    if isinstance(window, dict):
        return window[name]
    return getattr(window, name)


def _validated_windows(binding):
    windows = binding.get("windows") if isinstance(binding, dict) else binding
    windows = tuple(windows)
    if len(windows) != EXPECTED_REAL_VIALS:
        raise ValueError(f"real check needs {EXPECTED_REAL_VIALS} windows; got {len(windows)}")
    if sum(int(_window_value(item, "frame_count")) for item in windows) != EXPECTED_REAL_FRAMES:
        raise ValueError("real-check windows must contain exactly 1,470 frames")
    return windows


def _decoded_pair(value):
    if isinstance(value, dict):
        if "rgb" in value:
            return int(value["frame_index"]), value["rgb"]
        if "bgr" in value:
            return int(value["frame_index"]), real_eval.bgr_to_rgb(value["bgr"])
        return int(value["frame_index"]), value["frame"]
    frame_index, frame = value
    return int(frame_index), frame


def _video_metadata(capture, cv2):
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the real-check video")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (width, height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(f"real-check video must be 640x480; got {width}x{height}")
    if frame_count <= LAST_REQUIRED_FRAME:
        raise ValueError("real-check video does not contain the last approved window frame")
    if not np.isfinite(fps) or not 29.0 <= fps <= 31.0:
        raise ValueError(f"real-check video must be about 30 fps; got {fps}")
    return {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "fps": fps,
    }


def _infer_window(torch, model, device, decoded, batch_size, crop_xyxy, rule_options):
    prepared = []
    for value in decoded:
        frame_index, rgb = _decoded_pair(value)
        if rgb.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
            raise ValueError(
                f"frame {frame_index} must be 640x480 RGB; got {tuple(rgb.shape)}"
            )
        x0, y0, x1, y1 = (int(item) for item in crop_xyxy)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float().div_(255.0)
        prepared.append((frame_index, tensor))

    results = []
    with torch.inference_mode():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            images = torch.stack([item[1] for item in batch]).to(
                device, non_blocking=True
            )
            masks = model(images).argmax(dim=1).cpu().numpy()
            for (frame_index, _tensor), mask in zip(batch, masks, strict=True):
                prediction = surface_eval.surface_prediction(mask, **rule_options)
                results.append({"frame_index": frame_index, **prediction})
    return results


def run(args):
    # These gates read only frozen metadata/model inputs and video bytes, never video frames.
    authority = test_eval.validate_authority(
        batch_summary_path=args.batch_summary,
        batch_summary_sha256=args.batch_summary_sha256,
        training_summary_path=args.training_summary,
        training_summary_sha256=args.training_summary_sha256,
        checkpoint_path=args.checkpoint,
        weights_sha256=args.weights_sha256,
        training_module_sha256=args.training_module_sha256,
        frozen_rule_path=args.frozen_rule,
        frozen_rule_sha256=args.frozen_rule_sha256,
        frozen_ruler_path=args.frozen_ruler,
        frozen_ruler_sha256=args.frozen_ruler_sha256,
    )
    real_binding = real_eval.validate_real_inputs(
        args.windows,
        args.windows_sha256,
        args.recipes,
        args.recipes_sha256,
    )
    windows = _validated_windows(real_binding)
    if args.video_sha256 != EXPECTED_VIDEO_SHA256:
        raise ValueError("video SHA-256 argument is not the frozen real-check source")
    observed_video_hash = training.file_sha256(args.video)
    if observed_video_hash != EXPECTED_VIDEO_SHA256:
        raise ValueError("real-check video SHA-256 does not match the frozen source")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_root}")

    import cv2
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    training.seed_everything(authority["model"]["training_seed"])
    model = _load_model(torch, args.checkpoint, authority["model"], device)

    # Output creation and VideoCapture happen only after every immutable input/model gate.
    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "state.jsonl"
    frame_path = args.output_root / "real-frames.jsonl"
    _append_state(state_path, "RUNNING")
    started = time.time()
    decoded_frames = 0
    completed_vials = 0
    capture = None
    try:
        capture = cv2.VideoCapture(str(args.video))
        video_metadata = _video_metadata(capture, cv2)
        per_vial = []
        with frame_path.open("x", encoding="utf-8") as frame_handle:
            for window in windows:
                try:
                    decoded = real_eval.decode_window(capture, window, cv2)
                except Exception as error:
                    decoded_frames += int(getattr(error, "frames_decoded", 0))
                    raise
                decoded_frames += len(decoded)
                frame_results = _infer_window(
                    torch,
                    model,
                    device,
                    decoded,
                    args.batch_size,
                    authority["model"]["crop_xyxy"],
                    authority["rule"]["rule_options"],
                )
                vial = int(_window_value(window, "vial"))
                for result in frame_results:
                    frame_handle.write(
                        json.dumps(
                            {
                                "vial": vial,
                                "zone": _window_value(window, "zone"),
                                "transition_or_repeat": _window_value(
                                    window, "transition_or_repeat"
                                ),
                                **result,
                            },
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                per_vial.append(
                    real_eval.aggregate_vial(
                        window,
                        frame_results,
                        authority["ruler"]["ruler"],
                    )
                )
                completed_vials += 1
        if decoded_frames != EXPECTED_REAL_FRAMES:
            raise RuntimeError(
                f"real check decoded {decoded_frames} frames; expected {EXPECTED_REAL_FRAMES}"
            )
        if completed_vials != EXPECTED_REAL_VIALS:
            raise RuntimeError(
                f"real check completed {completed_vials} vials; expected {EXPECTED_REAL_VIALS}"
            )
        aggregate = real_eval.summarize_vials(per_vial)
        frame_evidence_hash = training.file_sha256(frame_path)
        summary = {
            "schema_version": 1,
            "purpose": "Step 10 final-volume real-check evaluation",
            "status": "complete",
            "evidence_label": "measured_on_27_reserved_real_check_vials",
            "checkpoint_review_sha256": test_eval.EXPECTED_CHECKPOINT_REVIEW_SHA256,
            "disclosures": {
                "system_calibration": "failed",
                "continued_after_failed_calibration": "explicit_user_decision",
                "ruler": "hand_measured_step3_targets_not_system_fitted",
                "result_use": "measured_once_and_untuned",
            },
            "dataset": {
                "role": "real_check",
                "vial_count": len(per_vial),
                "vials": [int(_window_value(item, "vial")) for item in windows],
                "zone_counts": dict(real_binding["zone_counts"]),
                "transition_or_repeat_counts": dict(
                    real_binding["transition_or_repeat_counts"]
                ),
                "excluded_vials": dict(real_binding["excluded_vials"]),
                "settled_frames_loaded": decoded_frames,
                "vial_windows_decoded": len(windows),
                "windows_csv_opened": 1,
                "recipes_csv_opened": 1,
                "video_files_hashed": 1,
                "train_manifests_opened": 0,
                "train_images_loaded": 0,
                "val_manifests_opened": 0,
                "val_images_loaded": 0,
                "test_manifests_opened": 0,
                "test_images_loaded": 0,
                "calibration_manifests_opened": 0,
                "calibration_frames_loaded": 0,
                "real_check_frames_loaded": decoded_frames,
                "video": video_metadata,
            },
            "project_access_context": {
                "prior_step9_test_manifests_opened": 4,
                "prior_step9_test_images_loaded_once": 560,
                "step10_test_images_loaded": 0,
            },
            "aggregate": aggregate,
            "per_vial": per_vial,
            "frozen": {
                "rule_config": authority["rule"]["rule_config"],
                "ruler": authority["ruler"]["ruler"],
                "ruler_source": authority["ruler"]["source"],
            },
            "model": {
                **authority["model"],
                "device": str(device),
                "torch_version": torch.__version__,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            },
            "inputs": {
                "windows_sha256": args.windows_sha256,
                "recipes_sha256": args.recipes_sha256,
                "video_sha256": observed_video_hash,
                "batch_summary_sha256": authority["batch"]["artifact_sha256"],
                "training_summary_sha256": authority["model"][
                    "training_summary_sha256"
                ],
                "weights_sha256": authority["model"]["weights_sha256"],
                "training_module_sha256": authority["model"][
                    "training_module_sha256"
                ],
                "frozen_rule_sha256": authority["rule"]["artifact_sha256"],
                "surface_rule_module_sha256": authority["rule"][
                    "surface_module_sha256"
                ],
                "frozen_ruler_sha256": authority["ruler"]["artifact_sha256"],
                "ruler_module_sha256": authority["ruler"]["ruler_module_sha256"],
                "real_evaluator_module_sha256": training.file_sha256(real_eval.__file__),
                "evaluator_script_sha256": training.file_sha256(__file__),
            },
            "frame_evidence": {
                "filename": frame_path.name,
                "sha256": frame_evidence_hash,
                "line_count": decoded_frames,
            },
            "elapsed_seconds": time.time() - started,
        }
        full_path = args.output_root / "summary.json"
        _json_dump(full_path, summary)
        compact = dict(summary)
        compact["full_summary_sha256"] = training.file_sha256(full_path)
        compact_path = args.output_root / "compact-summary.json"
        _json_dump(compact_path, compact)
        _append_state(
            state_path,
            "COMPLETE",
            real_check_frames_loaded=decoded_frames,
            full_summary_sha256=training.file_sha256(full_path),
            compact_summary_sha256=training.file_sha256(compact_path),
            frame_evidence_sha256=frame_evidence_hash,
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "vials": len(per_vial),
                    "real_check_frames": decoded_frames,
                    "compact_summary_sha256": training.file_sha256(compact_path),
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return summary
    except Exception as error:
        failure_path = args.output_root / "failure.json"
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "real_check_frames_decoded": decoded_frames,
            "vials_completed": completed_vials,
        }
        if frame_path.exists():
            failure["frame_evidence_sha256"] = training.file_sha256(frame_path)
        _json_dump(failure_path, failure)
        _append_state(
            state_path,
            "FAILED",
            real_check_frames_decoded=decoded_frames,
            vials_completed=completed_vials,
            failure_sha256=training.file_sha256(failure_path),
        )
        raise
    finally:
        if capture is not None:
            capture.release()


def main():
    run(_parser().parse_args())


if __name__ == "__main__":
    main()

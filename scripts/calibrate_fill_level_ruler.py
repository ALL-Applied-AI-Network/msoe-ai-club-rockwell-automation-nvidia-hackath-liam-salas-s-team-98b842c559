#!/usr/bin/env python3
"""Fit the frozen Step 8 FillCam ruler from settled vials 1 and 13.

Every hash-indexed frame in the two committed calibration windows is inferred.
The validation-frozen surface rule is applied unchanged, visible rows are
median-aggregated per vial, and the two calibration points define the affine
ruler before any held-out twin render is opened.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import traceback

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix import fill_level_ruler as ruler  # noqa: E402
from fluidmix import fill_level_surface_eval as surface_eval  # noqa: E402
from fluidmix import fill_level_training as training  # noqa: E402


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_IMAGE_SIZE = (640, 480)
EXPECTED_VIDEO_SHA256 = "6f87cec1aeec3b2bae6dac639a317c077a3f175f27abd5fc8430d6953d9d05b9"
EXPECTED_WINDOWS_SHA256 = "19174a4b706fe896b177ba71c8ba2bdf850ceb9cf3a40522dcdaa42d53904229"
LOGICAL_VIDEO_PATH = "csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4"
LOGICAL_WINDOWS_PATH = "data/real_check_windows.csv"
EXPECTED_RECIPES = {1: "5", 13: "9"}
EXPECTED_VALIDATION_CASE_COUNTS = {
    "clean": 69,
    "foam": 142,
    "sticker": 172,
    "edge": 153,
    "hidden": 72,
}
REQUIRED_VALIDATION_AGGREGATE_KEYS = {
    "frames",
    "visible_predictions",
    "visible_prediction_rate",
    "abstained",
    "abstention_rate",
    "predicted_hidden",
    "predicted_hidden_rate",
    "geometry_in_predicted_range",
    "true_in_range_rate",
    "true_in_range_over_all_frames_rate",
    "mae_rows",
    "median_absolute_error_rows",
    "p95_absolute_error_rows",
    "max_absolute_error_rows",
    "mae_ml",
    "median_absolute_error_ml",
    "p95_absolute_error_ml",
    "max_absolute_error_ml",
    "hidden_interval_median_distance_to_label_rows",
    "hidden_interval_p95_distance_to_label_rows",
    "hidden_interval_max_distance_to_label_rows",
    "hidden_interval_median_distance_to_label_ml",
    "hidden_interval_p95_distance_to_label_ml",
    "hidden_interval_max_distance_to_label_ml",
}


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _sha256(value, description):
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise ValueError(f"{description} must be a SHA-256")
    return value.lower()


def _safe_relative_file(root, value, description):
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {description} path: {value}")
    root = Path(root).resolve()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"unsafe {description} path: {value}")
    if not resolved.is_file():
        raise ValueError(f"{description} file does not exist: {relative.as_posix()}")
    return relative.as_posix(), resolved


def validate_frame_index(index_path):
    """Validate the extractor index without trusting absolute provenance paths."""

    index_path = Path(index_path)
    value = _read_json(index_path, "calibration frame index")
    if value.get("schema_version") != 1:
        raise ValueError("unsupported calibration frame-index schema")
    if not str(value.get("selection_policy", "")).startswith("all frames"):
        raise ValueError("calibration index must use all committed window frames")
    image_contract = value.get("image_contract")
    if not isinstance(image_contract, dict) or (
        image_contract.get("encoding") != "PNG"
        or image_contract.get("lossless") is not True
        or (int(image_contract.get("width", 0)), int(image_contract.get("height", 0)))
        != EXPECTED_IMAGE_SIZE
    ):
        raise ValueError("calibration index must declare lossless 640x480 PNG frames")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("calibration index has no source provenance")
    video = source.get("video")
    windows_source = source.get("windows_csv")
    if not isinstance(video, dict) or not isinstance(windows_source, dict):
        raise ValueError("calibration source provenance is incomplete")
    video_hash = _sha256(video.get("sha256"), "source video hash")
    if video_hash != EXPECTED_VIDEO_SHA256:
        raise ValueError("calibration index is not from the approved video-5 bytes")
    windows_hash = _sha256(windows_source.get("sha256"), "windows CSV hash")
    if windows_hash != EXPECTED_WINDOWS_SHA256:
        raise ValueError("calibration index is not from the committed settled windows")

    windows = value.get("windows")
    frames = value.get("frames")
    if not isinstance(windows, list) or not isinstance(frames, list):
        raise ValueError("calibration index needs windows and frames lists")
    if len(windows) != 2:
        raise ValueError("calibration index must contain exactly two windows")
    by_vial = {}
    for window in windows:
        if not isinstance(window, dict):
            raise ValueError("calibration window must be an object")
        vial = int(window.get("vial"))
        if vial in by_vial:
            raise ValueError(f"calibration index repeats vial {vial}")
        if vial not in ruler.CALIBRATION_VOLUMES_ML:
            raise ValueError(f"unexpected calibration vial {vial}")
        if str(window.get("recipe_id")) != EXPECTED_RECIPES[vial]:
            raise ValueError(f"vial {vial} recipe does not match the frozen calibration")
        if window.get("role") != "calibration" or window.get("source_role") != "real_validation":
            raise ValueError(f"vial {vial} roles do not match the calibration contract")
        volume_ml = float(window.get("level_ml"))
        if volume_ml != ruler.CALIBRATION_VOLUMES_ML[vial]:
            raise ValueError(f"vial {vial} calibration volume is incorrect")
        start = int(window.get("start_frame"))
        end = int(window.get("end_frame"))
        if (start, end) != ruler.CALIBRATION_FRAME_SPANS[vial]:
            raise ValueError(f"vial {vial} frame span is not the committed settled window")
        if int(window.get("n_frames")) != end - start + 1:
            raise ValueError(f"vial {vial} window frame count is inconsistent")
        by_vial[vial] = {
            "vial": vial,
            "volume_ml": volume_ml,
            "start_frame": start,
            "end_frame": end,
            "frames": [],
        }
    if set(by_vial) != set(ruler.CALIBRATION_VOLUMES_ML):
        raise ValueError("calibration index must contain vials 1 and 13")

    seen_paths = set()
    seen_keys = set()
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("calibration frame must be an object")
        vial = int(frame.get("vial"))
        if vial not in by_vial:
            raise ValueError(f"frame references unexpected vial {vial}")
        number = int(frame.get("frame_index"))
        key = (vial, number)
        if key in seen_keys:
            raise ValueError(f"calibration index repeats vial/frame {key}")
        seen_keys.add(key)
        if str(frame.get("recipe_id")) != EXPECTED_RECIPES[vial]:
            raise ValueError(f"vial {vial} frame recipe is incorrect")
        if frame.get("role") != "calibration" or frame.get("source_window_role") != "real_validation":
            raise ValueError(f"vial {vial} frame roles are incorrect")
        if float(frame.get("level_ml")) != ruler.CALIBRATION_VOLUMES_ML[vial]:
            raise ValueError(f"vial {vial} frame volume is incorrect")
        width, height = int(frame.get("width")), int(frame.get("height"))
        if (width, height) != EXPECTED_IMAGE_SIZE:
            raise ValueError(f"frame {number} is not 640x480")
        portable, resolved = _safe_relative_file(
            index_path.parent, frame.get("path"), "calibration frame"
        )
        if portable in seen_paths:
            raise ValueError(f"calibration index repeats path {portable}")
        seen_paths.add(portable)
        by_vial[vial]["frames"].append(
            {
                "frame": number,
                "path": portable,
                "resolved_path": resolved,
                "sha256": _sha256(frame.get("sha256"), "calibration frame hash"),
            }
        )
    normalized = []
    for vial in sorted(by_vial):
        item = by_vial[vial]
        item["frames"].sort(key=lambda frame: frame["frame"])
        expected = list(range(item["start_frame"], item["end_frame"] + 1))
        if [frame["frame"] for frame in item["frames"]] != expected:
            raise ValueError(f"vial {vial} does not contain every committed window frame")
        normalized.append(item)
    if int(value.get("frame_count")) != sum(len(item["frames"]) for item in normalized):
        raise ValueError("calibration frame_count does not match indexed records")
    return {
        "index_sha256": training.file_sha256(index_path),
        "source_video": {"logical_path": LOGICAL_VIDEO_PATH, "sha256": video_hash},
        "windows": {"path": LOGICAL_WINDOWS_PATH, "sha256": windows_hash},
        "vials": normalized,
        "frame_count": sum(len(item["frames"]) for item in normalized),
    }


def validate_frozen_rule(rule_path, expected_rule_sha256, checkpoint_sha256):
    """Load the complete validation artifact and extract its immutable rule."""

    rule_path = Path(rule_path)
    observed_rule_hash = training.file_sha256(rule_path)
    if observed_rule_hash != expected_rule_sha256:
        raise ValueError("validation rule artifact SHA-256 does not match the frozen hash")
    value = _read_json(rule_path, "frozen validation rule")
    if value.get("status") not in ("complete", "frozen_before_test"):
        raise ValueError("validation rule artifact must be complete")
    config = value.get("rule_config")
    if not isinstance(config, dict):
        raise ValueError("validation rule artifact has no rule_config")
    if config.get("selection_source") != "synthetic_validation_only":
        raise ValueError("surface rule must be selected on synthetic validation only")
    if config.get("crop_xyxy") != list(training.CROP_XYXY):
        raise ValueError("surface rule crop does not match the trained model crop")
    if config.get("class_ids") != {"liquid": 1, "blocked": 3}:
        raise ValueError("surface rule class ids do not match the trained model")
    columns = config.get("measurement_columns_full_half_open")
    if not isinstance(columns, list) or len(columns) != 2:
        raise ValueError("surface rule measurement columns must contain two values")
    try:
        options = {
            "measurement_columns_full": tuple(int(value) for value in columns),
            "min_liquid_pixels": int(config["minimum_liquid_pixels_per_row"]),
            "min_blocked_pixels": int(config["minimum_blocked_pixels_per_row"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("surface rule config is incomplete") from error
    # Exercise the public rule's own bounds validation without opening an image.
    surface_eval.surface_prediction(
        np.zeros(
            (
                training.CROP_XYXY[3] - training.CROP_XYXY[1],
                training.CROP_XYXY[2] - training.CROP_XYXY[0],
            ),
            dtype=np.uint8,
        ),
        **options,
    )
    dataset = value.get("dataset")
    required_safety = {
        "role": "val",
        "frame_count": 300,
        "case_counts": EXPECTED_VALIDATION_CASE_COUNTS,
        "train_manifests_opened": 0,
        "train_images_loaded": 0,
        "test_manifests_opened": 0,
        "test_images_loaded": 0,
        "real_frames_loaded": 0,
    }
    if not isinstance(dataset, dict) or any(
        dataset.get(key) != expected for key, expected in required_safety.items()
    ):
        raise ValueError(
            "surface rule artifact is not the complete leak-free 300-frame validation run"
        )

    aggregate = value.get("aggregate")
    by_case = aggregate.get("by_case") if isinstance(aggregate, dict) else None
    by_bin = aggregate.get("by_level_bin_10ml") if isinstance(aggregate, dict) else None
    expected_cases = {"all"} | set(EXPECTED_VALIDATION_CASE_COUNTS)
    expected_bins = {
        f"{lower:02d}-{lower + 10:02d}" for lower in range(0, 90, 10)
    }
    if not isinstance(by_case, dict) or set(by_case) != expected_cases:
        raise ValueError("surface rule artifact validation case aggregates are incomplete")
    if not isinstance(by_bin, dict) or set(by_bin) != expected_bins:
        raise ValueError("surface rule artifact validation level-bin aggregates are incomplete")
    for group_name, group in (("case", by_case), ("level-bin", by_bin)):
        for name, metrics in group.items():
            missing = (
                REQUIRED_VALIDATION_AGGREGATE_KEYS - set(metrics)
                if isinstance(metrics, dict)
                else REQUIRED_VALIDATION_AGGREGATE_KEYS
            )
            if missing:
                raise ValueError(
                    f"surface rule artifact validation {group_name} {name} "
                    f"metrics are incomplete: {sorted(missing)}"
                )
    try:
        if int(by_case["all"]["frames"]) != 300:
            raise ValueError
        for case, expected in EXPECTED_VALIDATION_CASE_COUNTS.items():
            if int(by_case[case]["frames"]) != expected:
                raise ValueError
        if sum(int(by_bin[name]["frames"]) for name in expected_bins) != 300:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "surface rule artifact validation aggregate frame counts are invalid"
        ) from None

    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("checkpoint_sha256"), str
    ):
        raise ValueError("surface rule artifact has no checkpoint hash binding")
    recorded = inputs["checkpoint_sha256"]
    if recorded != checkpoint_sha256:
        raise ValueError("surface rule checkpoint hash does not match calibration checkpoint")
    return {
        "artifact_sha256": observed_rule_hash,
        "rule_config": config,
        "rule_options": options,
    }


def _append_state(path, status, **details):
    record = {
        "status": status,
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **details,
    }
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _load_model(torch, checkpoint_path, checkpoint_sha256, device):
    observed = training.file_sha256(checkpoint_path)
    if observed != checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match --weights-sha256")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported training checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("training checkpoint has no config")
    if config.get("crop_xyxy") != list(training.CROP_XYXY):
        raise ValueError("training checkpoint crop does not match calibration crop")
    expected_classes = {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3}
    if config.get("class_ids") != expected_classes:
        raise ValueError("training checkpoint class ids do not match the row rule")
    base_channels = int(config.get("base_channels"))
    model = training.build_unet(base_channels=base_channels)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, {
        "weights_sha256": observed,
        "weights_filename": Path(checkpoint_path).name,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_channels": base_channels,
        "crop_xyxy": list(training.CROP_XYXY),
        "class_ids": expected_classes,
    }


def _load_rgb(frame):
    observed_hash = training.file_sha256(frame["resolved_path"])
    if observed_hash != frame["sha256"]:
        raise ValueError(f"frame {frame['frame']} SHA-256 does not match its index")
    with Image.open(frame["resolved_path"]) as image:
        if image.size != EXPECTED_IMAGE_SIZE:
            raise ValueError(f"frame {frame['frame']} decoded dimensions are not 640x480")
        rgb = np.asarray(image.convert("RGB")).copy()
    x0, y0, x1, y1 = training.CROP_XYXY
    return np.ascontiguousarray(rgb[y0:y1, x0:x1])


def infer_calibration_frames(torch, model, device, indexed, rule_options, batch_size):
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    flattened = [
        (vial["vial"], vial["volume_ml"], frame)
        for vial in indexed["vials"]
        for frame in vial["frames"]
    ]
    records = []
    with torch.inference_mode():
        for first in range(0, len(flattened), batch_size):
            selected = flattened[first : first + batch_size]
            arrays = [_load_rgb(frame) for _vial, _volume, frame in selected]
            tensor = torch.from_numpy(
                np.stack([array.transpose(2, 0, 1) for array in arrays])
            ).float().div_(255.0).to(device)
            predicted = model(tensor).argmax(dim=1).cpu().numpy()
            for (vial, volume_ml, frame), mask in zip(selected, predicted):
                prediction = surface_eval.surface_prediction(mask, **rule_options)
                records.append(
                    {
                        "vial": vial,
                        "volume_ml": volume_ml,
                        "frame": frame["frame"],
                        "frame_path": frame["path"],
                        "frame_sha256": frame["sha256"],
                        **prediction,
                    }
                )
    return records


def run(args):
    checkpoint_hash = _sha256(args.weights_sha256, "weights hash")
    expected_rule_hash = _sha256(
        args.frozen_rule_sha256, "frozen validation rule hash"
    )
    # All gates run before the fresh output directory is created.
    frozen_rule = validate_frozen_rule(
        args.frozen_rule, expected_rule_hash, checkpoint_hash
    )
    indexed = validate_frame_index(args.frames_index)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    state_path = args.output_root / "state.jsonl"
    _append_state(state_path, "RUNNING")
    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        device = torch.device(args.device)
        model, model_evidence = _load_model(
            torch, args.weights, checkpoint_hash, device
        )
        records = infer_calibration_frames(
            torch,
            model,
            device,
            indexed,
            frozen_rule["rule_options"],
            args.batch_size,
        )
        per_frame_path = args.output_root / "calibration-frames.jsonl"
        with per_frame_path.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

        calibration = []
        rows = {}
        for vial in indexed["vials"]:
            selected = [record for record in records if record["vial"] == vial["vial"]]
            row_summary = ruler.median_visible_row(selected)
            rows[vial["vial"]] = row_summary["median_row"]
            calibration.append(
                {
                    "vial": vial["vial"],
                    "volume_ml": vial["volume_ml"],
                    "start_frame": vial["start_frame"],
                    "end_frame": vial["end_frame"],
                    **row_summary,
                }
            )
        fitted = ruler.fit_affine_ruler(rows)
        summary = {
            "schema_version": 1,
            "purpose": "FillCam Step 8 real calibration and frozen row-to-volume ruler",
            "status": "complete",
            "freeze_state": "frozen_before_twin_test",
            "selection_policy": (
                "all indexed settled-window frames; median of visible frozen-rule rows; "
                "no appearance-based selection"
            ),
            "inputs": {
                "frame_index_sha256": indexed["index_sha256"],
                "frozen_validation_rule_sha256": frozen_rule["artifact_sha256"],
                "validation_summary_sha256": frozen_rule["artifact_sha256"],
                "weights_sha256": checkpoint_hash,
                "calibrator_script_sha256": training.file_sha256(__file__),
                "ruler_module_sha256": training.file_sha256(ruler.__file__),
                "surface_rule_module_sha256": training.file_sha256(
                    surface_eval.__file__
                ),
                "training_module_sha256": training.file_sha256(training.__file__),
                "test_manifests_opened": 0,
                "test_images_loaded": 0,
            },
            "source_video": indexed["source_video"],
            "windows": indexed["windows"],
            "frame_count": indexed["frame_count"],
            "rule_config": frozen_rule["rule_config"],
            "model": model_evidence,
            "calibration": calibration,
            "ruler": fitted,
            "artifacts": {
                "per_frame": per_frame_path.name,
                "per_frame_sha256": training.file_sha256(per_frame_path),
            },
        }
        summary_path = args.output_root / "ruler-summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _append_state(
            state_path,
            "COMPLETE",
            summary_sha256=training.file_sha256(summary_path),
        )
        return summary
    except Exception as error:
        _append_state(
            state_path,
            "FAILED",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-index", type=Path, required=True)
    parser.add_argument("--frozen-rule", type=Path, required=True)
    parser.add_argument("--frozen-rule-sha256", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

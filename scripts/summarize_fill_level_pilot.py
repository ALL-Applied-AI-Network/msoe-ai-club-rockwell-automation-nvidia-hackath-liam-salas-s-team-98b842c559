"""Summarize a copied FillCam pilot without putting raw frames in Git.

The input directory must contain ``manifest.jsonl`` and ``summary.json`` from
``render_fill_level_dataset.py``.  The report contains measurements and source
hashes, not image pixels.  The optional visual montage is written only to the
explicit path supplied by the caller (normally below ignored ``outputs/``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import check_fill_level_capture as capture_checker  # noqa: E402


LABEL_NAMES = {
    0: "empty",
    1: "liquid",
    2: "foam",
    3: "blocked",
    255: "ignore",
}
MASK_COLORS_BGR = {
    0: (190, 190, 190),
    1: (220, 105, 35),
    2: (80, 230, 250),
    3: (210, 45, 210),
    255: (0, 0, 0),
}
HELD_OUT_TARGET_ROWS = {70.0: 206.0, 75.0: 194.0}
STEP6_TARGET_COUNTS = {"train": 3000, "val": 300, "test": 500}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_span(text):
    """Parse a half-open START:STOP pixel interval."""
    start_text, separator, stop_text = str(text).partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("pixel spans use START:STOP")
    try:
        start, stop = int(start_text), int(stop_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pixel span bounds must be integers") from error
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("pixel spans require 0 <= START < STOP")
    return start, stop


def _read_records(manifest_path):
    records = []
    for line_number, raw_line in enumerate(manifest_path.read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"manifest line {line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"manifest line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError("manifest contains no frame records")
    return records


def _output_path(manifest_path, record, kind):
    try:
        raw_path = record["outputs"][kind]["path"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"record is missing outputs.{kind}.path") from error
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def measure_vial_edges(image, rows, left_window, right_window):
    """Measure dark vial boundaries using mean horizontal grayscale gradients."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("edge measurement requires grayscale or BGR image pixels")
    height, width = gray.shape
    row_start, row_stop = rows
    if not (0 <= row_start < row_stop <= height):
        raise ValueError(f"row span {rows} is outside image height {height}")
    for name, (start, stop) in (
        ("left", left_window),
        ("right", right_window),
    ):
        if not (0 <= start < stop <= width) or stop - start < 2:
            raise ValueError(f"{name} window {(start, stop)} is invalid for width {width}")
    profile = gray[row_start:row_stop].astype(np.float64).mean(axis=0)
    gradient = np.abs(np.diff(profile))

    def strongest(window):
        start, stop = window
        relevant = gradient[start : stop - 1]
        index = int(np.argmax(relevant)) + start
        return {
            "x": index,
            "gradient": float(gradient[index]),
        }

    left = strongest(left_window)
    right = strongest(right_window)
    if right["x"] <= left["x"]:
        raise ValueError("right vial edge did not fall to the right of the left edge")
    return {
        "left": left,
        "right": right,
        "center_x": (left["x"] + right["x"]) / 2.0,
        "width_px": right["x"] - left["x"],
    }


def _timing_summary(records):
    keys = (
        "rgb",
        "glass_only",
        "glass_hidden_fluid",
        "interior_probe_blockers",
        "all_frame_passes",
    )
    result = {}
    for key in keys:
        values = np.asarray(
            [float(record["timings_seconds"][key]) for record in records],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"timing {key} must contain finite positive values")
        result[key] = {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    result["estimated_frames_per_hour_from_median_pass_time"] = (
        3600.0 / result["all_frame_passes"]["median"]
    )
    return result


def _batch_sizing(records, capture_summary, timing):
    process_seconds = float(capture_summary.get("seconds", 0.0))
    if not math.isfinite(process_seconds) or process_seconds <= 0.0:
        raise ValueError("render summary seconds must be finite and positive")
    phases = capture_summary.get("phase_timings_seconds", {})
    fixed_phase_seconds = sum(
        float(phases.get(name, 0.0))
        for name in ("simulation_app_start", "stage_load", "warmup")
    )
    steady_seconds = process_seconds - fixed_phase_seconds
    if not math.isfinite(steady_seconds) or steady_seconds <= 0.0:
        raise ValueError("startup-adjusted pilot seconds must be finite and positive")
    frame_count = len(records)
    end_to_end_frames_per_hour = 3600.0 * frame_count / process_seconds
    steady_frames_per_hour = 3600.0 * frame_count / steady_seconds
    target_total = sum(STEP6_TARGET_COUNTS.values())
    return {
        "selected_step6_counts": dict(STEP6_TARGET_COUNTS),
        "selected_total_frames": target_total,
        "measured_pilot_process_seconds": process_seconds,
        "measured_fixed_phase_seconds": fixed_phase_seconds,
        "end_to_end_frames_per_hour": end_to_end_frames_per_hour,
        "startup_adjusted_frames_per_hour": steady_frames_per_hour,
        "pass_time_frames_per_hour": timing[
            "estimated_frames_per_hour_from_median_pass_time"
        ],
        "estimated_hours_for_selected_total_conservative": (
            target_total / end_to_end_frames_per_hour
        ),
        "estimated_hours_for_selected_total_single_batch": (
            fixed_phase_seconds + 3600.0 * target_total / steady_frames_per_hour
        ) / 3600.0,
        "decision_basis": (
            "Keep the plan's full 3000/300/500 target. Report the 20-frame "
            "end-to-end rate as the conservative schedule and the startup-adjusted "
            "single-batch rate separately; pass-only throughput is diagnostic."
        ),
    }


def _held_out_rows(records):
    result = []
    for level_ml, real_target in HELD_OUT_TARGET_ROWS.items():
        matches = [
            record for record in records
            if math.isclose(float(record.get("level_ml", -1.0)), level_ml)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"pilot requires exactly one {level_ml:g} mL record; found {len(matches)}"
            )
        record = matches[0]
        geometry_row = record.get("surface", {}).get("geometry_row")
        rgb_row = record.get("surface", {}).get("rgb_row")
        measurement = {
            "level_ml": level_ml,
            "real_reference_row": real_target,
            "real_reference_note": (
                "about 206" if level_ml == 70.0 else "about 193-195; midpoint 194"
            ),
            "seed": int(record["seed"]),
            "geometry_row": float(geometry_row) if geometry_row is not None else None,
            "rgb_row": float(rgb_row) if rgb_row is not None else None,
        }
        for kind in ("geometry", "rgb"):
            value = measurement[f"{kind}_row"]
            measurement[f"{kind}_delta_px_vs_real"] = (
                value - real_target if value is not None else None
            )
        result.append(measurement)
    return result


def _label_coverage(check_result):
    frame_counts = {name: 0 for name in LABEL_NAMES.values()}
    pixel_counts = {name: 0 for name in LABEL_NAMES.values()}
    for report in check_result["records"]:
        counts = report.get("label_counts", {})
        for identifier, name in LABEL_NAMES.items():
            count = int(counts.get(str(identifier), 0))
            pixel_counts[name] += count
            frame_counts[name] += int(count > 0)
    return {
        "frames_with_class": frame_counts,
        "pixels_by_class": pixel_counts,
    }


def _case_coverage(records):
    levels = [float(record["level_ml"]) for record in records]
    holder_brightness = [
        float(record["holder_material"]["brightness"])
        for record in records if "holder_material" in record
    ]
    holder_roughness = [
        float(record["holder_material"]["roughness"])
        for record in records if "holder_material" in record
    ]
    return {
        "empty_frames": sum(level == 0.0 for level in levels),
        "foam_enabled_frames": sum(
            bool(record.get("foam", {}).get("enabled")) for record in records
        ),
        "sticker_present_frames": sum(
            bool(record.get("sticker_pose", {}).get("present")) for record in records
        ),
        "wet_wall_frames": sum(
            bool(record.get("wet_walls", {}).get("enabled")) for record in records
        ),
        "twin_lower_edge_frames": sum(40.1 <= level < 48.1 for level in levels),
        "twin_hidden_frames": sum(48.1 <= level < 66.5 for level in levels),
        "twin_upper_edge_frames": sum(66.5 <= level <= 70.5 for level in levels),
        "holder_material": {
            "recorded_frames": len(holder_brightness),
            "brightness_min": min(holder_brightness) if holder_brightness else None,
            "brightness_max": max(holder_brightness) if holder_brightness else None,
            "roughness_min": min(holder_roughness) if holder_roughness else None,
            "roughness_max": max(holder_roughness) if holder_roughness else None,
        },
    }


def _colorize_mask(mask):
    if mask.ndim != 2:
        raise ValueError("class mask must be two-dimensional")
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for identifier, color in MASK_COLORS_BGR.items():
        colored[mask == identifier] = color
    unexpected = ~np.isin(mask, list(MASK_COLORS_BGR))
    colored[unexpected] = (0, 0, 255)
    return colored


def write_montage(records, manifest_path, output_path, columns=4):
    """Write a compact RGB/mask side-by-side visual check."""
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"montage output already exists: {output_path}")
    if columns <= 0:
        raise ValueError("montage columns must be positive")
    image_width, image_height, title_height = 200, 150, 28
    tile_width = image_width * 2
    tile_height = image_height + title_height
    rows = math.ceil(len(records) / columns)
    canvas = np.full(
        (rows * tile_height, columns * tile_width, 3), 35, dtype=np.uint8
    )
    for index, record in enumerate(records):
        rgb_path = _output_path(manifest_path, record, "rgb")
        mask_path = _output_path(manifest_path, record, "mask")
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or mask is None:
            raise ValueError(f"could not decode montage inputs for record {index}")
        rgb_small = cv2.resize(
            rgb, (image_width, image_height), interpolation=cv2.INTER_AREA
        )
        mask_small = cv2.resize(
            _colorize_mask(mask),
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        row, column = divmod(index, columns)
        y0, x0 = row * tile_height, column * tile_width
        canvas[y0 + title_height : y0 + tile_height, x0 : x0 + image_width] = rgb_small
        canvas[
            y0 + title_height : y0 + tile_height,
            x0 + image_width : x0 + tile_width,
        ] = mask_small
        title = f"seed {record['seed']}  {float(record['level_ml']):g} mL  RGB | mask"
        cv2.putText(
            canvas,
            title,
            (x0 + 5, y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"could not write montage: {output_path}")
    return output_path


def build_report(
    pilot_dir,
    real_still,
    position_seed,
    edge_rows=(90, 160),
    left_window=(260, 300),
    right_window=(320, 360),
):
    pilot_dir = Path(pilot_dir).expanduser().resolve()
    manifest_path = pilot_dir / "manifest.jsonl"
    summary_path = pilot_dir / "summary.json"
    real_still = Path(real_still).expanduser().resolve()
    for name, path in (
        ("pilot manifest", manifest_path),
        ("pilot summary", summary_path),
        ("real still", real_still),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    records = _read_records(manifest_path)
    capture_summary = json.loads(summary_path.read_text())
    check_result = capture_checker.validate_manifest(manifest_path)
    manifest_hash = file_sha256(manifest_path)
    summary_manifest_hash = capture_summary.get("manifest_sha256")
    summary_matches_manifest = (
        isinstance(summary_manifest_hash, str)
        and summary_manifest_hash.casefold() == manifest_hash.casefold()
    )
    integrity_errors = list(check_result["errors"])
    if not summary_matches_manifest:
        integrity_errors.append(
            "render summary manifest_sha256 does not match the analyzed manifest"
        )
    position_matches = [
        record for record in records if int(record.get("seed", -1)) == position_seed
    ]
    if len(position_matches) != 1:
        raise ValueError(
            f"position seed {position_seed} must identify one record; "
            f"found {len(position_matches)}"
        )
    position_record = position_matches[0]
    synthetic_path = _output_path(manifest_path, position_record, "rgb")
    synthetic = cv2.imread(str(synthetic_path), cv2.IMREAD_COLOR)
    real = cv2.imread(str(real_still), cv2.IMREAD_COLOR)
    if synthetic is None or real is None:
        raise ValueError("could not decode the synthetic position frame or real still")
    synthetic_edges = measure_vial_edges(
        synthetic, edge_rows, left_window, right_window
    )
    real_edges = measure_vial_edges(real, edge_rows, left_window, right_window)
    timing = _timing_summary(records)
    summary_frame_count = int(capture_summary.get("frame_count", -1))
    summary_passed = bool(capture_summary.get("passed"))
    return {
        "schema_version": 1,
        "purpose": "FillCam step 5 pilot measurements",
        "evidence_label": "measured synthetic pilot and real-reference geometry",
        "sources": {
            "manifest_sha256": manifest_hash,
            "render_summary_sha256": file_sha256(summary_path),
            "real_still_basename": real_still.name,
            "real_still_sha256": file_sha256(real_still),
        },
        "renderer": {
            "name": capture_summary.get("renderer"),
            "subframes": capture_summary.get("subframes"),
            "resolution": capture_summary.get("resolution"),
        },
        "frame_count": len(records),
        "integrity": {
            "passed": bool(
                check_result["passed"]
                and summary_passed
                and summary_frame_count == len(records)
                and summary_matches_manifest
            ),
            "offline_capture_check_passed": bool(check_result["passed"]),
            "render_summary_passed": summary_passed,
            "render_summary_frame_count": summary_frame_count,
            "render_summary_matches_manifest": summary_matches_manifest,
            "errors": integrity_errors,
        },
        "case_coverage": _case_coverage(records),
        "label_coverage": _label_coverage(check_result),
        "held_out_surface_rows": _held_out_rows(records),
        "horizontal_vial_position": {
            "method": "largest absolute x-gradient of mean grayscale profile",
            "rows_half_open": list(edge_rows),
            "left_window_half_open": list(left_window),
            "right_window_half_open": list(right_window),
            "synthetic": {
                "seed": position_seed,
                "level_ml": float(position_record["level_ml"]),
                "rgb_basename": synthetic_path.name,
                **synthetic_edges,
            },
            "real": {
                "still_basename": real_still.name,
                **real_edges,
            },
            "delta_px_synthetic_minus_real": {
                "left": synthetic_edges["left"]["x"] - real_edges["left"]["x"],
                "right": synthetic_edges["right"]["x"] - real_edges["right"]["x"],
                "center": synthetic_edges["center_x"] - real_edges["center_x"],
                "width": synthetic_edges["width_px"] - real_edges["width_px"],
            },
        },
        "timing_seconds": timing,
        "batch_sizing": _batch_sizing(records, capture_summary, timing),
    }


def _write_json_new(path, value):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"report output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--real-still", type=Path, required=True)
    parser.add_argument("--position-seed", type=int, default=8155)
    parser.add_argument("--edge-rows", type=parse_span, default=(90, 160))
    parser.add_argument("--left-window", type=parse_span, default=(260, 300))
    parser.add_argument("--right-window", type=parse_span, default=(320, 360))
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--montage-out", type=Path, required=True)
    parser.add_argument("--montage-columns", type=int, default=4)
    args = parser.parse_args()
    report = build_report(
        args.pilot_dir,
        args.real_still,
        args.position_seed,
        args.edge_rows,
        args.left_window,
        args.right_window,
    )
    manifest_path = args.pilot_dir.expanduser().resolve() / "manifest.jsonl"
    records = _read_records(manifest_path)
    montage_path = write_montage(
        records, manifest_path, args.montage_out, args.montage_columns
    )
    report["visual_check_montage_basename"] = montage_path.name
    report_path = _write_json_new(args.report_out, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "montage": str(montage_path),
                "frames": report["frame_count"],
                "integrity_passed": report["integrity"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["integrity"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

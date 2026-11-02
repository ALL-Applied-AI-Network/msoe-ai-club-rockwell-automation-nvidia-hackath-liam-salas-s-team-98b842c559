"""Review retired visibility-rule exceptions before Step 7 training.

The full-batch summary intentionally inventories frames that would have failed
the original visibility rules.  This tool verifies each inventory row against
the copied manifest and source mask, creates ignored RGB/mask overlays, and
writes a compact manual-disposition report.  Source renders are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


LABEL_IDS = {"empty": 0, "liquid": 1, "foam": 2, "blocked": 3, "ignore": 255}
OVERLAY_BGR = {
    LABEL_IDS["empty"]: (255, 220, 0),
    LABEL_IDS["liquid"]: (40, 80, 255),
    LABEL_IDS["foam"]: (245, 245, 245),
    LABEL_IDS["blocked"]: (255, 0, 180),
}
APPROVED_DISPOSITIONS = {
    100327: {
        "disposition": "accepted_legitimate_occlusion",
        "explanation": (
            "The 0.065117 mL liquid is far below the visible base edge; visible foam "
            "and the visible sticker further occlude the vial. The decoded mask retains "
            "empty, foam, blocked and ignore pixels with no unexplained class loss."
        ),
        "required_factors": {
            "maximum_level_ml": 0.1,
            "foam_visible": True,
            "sticker_visible": True,
            "liquid_pixels": 0,
        },
    }
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _complete_attempt(chunk_dir):
    attempts = []
    for attempt in sorted(Path(chunk_dir).glob("attempt-*")):
        state_path = attempt / "state.txt"
        if state_path.is_file() and state_path.read_text(encoding="utf-8").strip() == "COMPLETE":
            attempts.append(attempt)
    if len(attempts) != 1:
        raise ValueError(
            f"{Path(chunk_dir).name} requires one COMPLETE attempt; found {len(attempts)}"
        )
    return attempts[0]


def _manifest_record(manifest_path, seed):
    matches = []
    for line_number, line in enumerate(
        Path(manifest_path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"manifest line {line_number} is invalid JSON: {error.msg}"
            ) from error
        if isinstance(record, dict) and record.get("seed") == seed:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"seed {seed} requires one manifest row; found {len(matches)}")
    return matches[0]


def _decoded_counts(mask):
    identifiers, counts = np.unique(mask, return_counts=True)
    measured = {int(identifier): int(count) for identifier, count in zip(identifiers, counts)}
    unexpected = set(measured) - set(LABEL_IDS.values())
    if unexpected:
        raise ValueError(f"mask has unexpected class ids: {sorted(unexpected)}")
    return {name: measured.get(identifier, 0) for name, identifier in LABEL_IDS.items()}


def _require_equal(actual, expected, description):
    if actual != expected:
        raise ValueError(f"{description} mismatch: expected {expected!r}, got {actual!r}")


def _verify_record(exception, record, output_dir):
    _require_equal(record.get("seed"), exception.get("seed"), "seed")
    _require_equal(record.get("role"), exception.get("role"), "role")
    _require_equal(record.get("frame_index"), exception.get("frame_index"), "frame index")
    _require_equal(
        record.get("level_sampling_band"),
        exception.get("sampling_band"),
        "sampling band",
    )
    if abs(float(record.get("level_ml")) - float(exception.get("level_ml"))) > 1e-12:
        raise ValueError("level_ml mismatch")

    outputs = record.get("outputs", {})
    rgb_info = outputs.get("rgb", {})
    mask_info = outputs.get("mask", {})
    rgb_path = output_dir / str(rgb_info.get("path", ""))
    mask_path = output_dir / str(mask_info.get("path", ""))
    if not rgb_path.is_file() or not mask_path.is_file():
        raise ValueError(f"seed {record.get('seed')} is missing its RGB or mask source")
    _require_equal(file_sha256(rgb_path), rgb_info.get("sha256"), "RGB SHA-256")
    _require_equal(file_sha256(mask_path), mask_info.get("sha256"), "mask SHA-256")

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or mask is None:
        raise ValueError(f"seed {record.get('seed')} RGB or mask could not be decoded")
    if mask.ndim != 2 or rgb.shape[:2] != mask.shape:
        raise ValueError(f"seed {record.get('seed')} RGB/mask shape mismatch")

    measured = _decoded_counts(mask)
    manifest_counts = record.get("label_stats", {}).get("class_pixels", {})
    _require_equal(measured, manifest_counts, "decoded mask class counts")
    expected_counts = exception.get("class_pixels", {})
    for name, value in expected_counts.items():
        _require_equal(measured.get(name), value, f"summary {name} pixels")

    foam = record.get("foam", {})
    summary_foam = exception.get("foam", {})
    for key in ("enabled", "visible_in_camera", "thickness_mm"):
        _require_equal(foam.get(key), summary_foam.get(key), f"foam {key}")
    sticker = record.get("sticker_pose", {})
    summary_sticker = exception.get("sticker", {})
    for key in ("present", "visible_in_camera"):
        _require_equal(sticker.get(key), summary_sticker.get(key), f"sticker {key}")
    _require_equal(
        record.get("label_stats", {}).get("sticker_blocked_pixels"),
        summary_sticker.get("blocked_pixels"),
        "sticker blocked pixels",
    )
    return rgb_path, mask_path, rgb, mask, measured


def _approved_disposition(exception, measured):
    seed = int(exception["seed"])
    approval = APPROVED_DISPOSITIONS.get(seed)
    if approval is None:
        return {
            "disposition": "excluded",
            "explanation": "No reviewed physical explanation is registered for this exception.",
        }
    factors = approval["required_factors"]
    valid = (
        float(exception["level_ml"]) <= factors["maximum_level_ml"]
        and bool(exception.get("foam", {}).get("visible_in_camera"))
        == factors["foam_visible"]
        and bool(exception.get("sticker", {}).get("visible_in_camera"))
        == factors["sticker_visible"]
        and measured["liquid"] == factors["liquid_pixels"]
    )
    if not valid:
        return {
            "disposition": "excluded",
            "explanation": "The registered explanation did not match the verified frame evidence.",
        }
    return {
        "disposition": approval["disposition"],
        "explanation": approval["explanation"],
    }


def _overlay_image(rgb, mask):
    result = rgb.copy()
    tint = np.zeros_like(rgb)
    scored = np.zeros(mask.shape, dtype=bool)
    for identifier, color in OVERLAY_BGR.items():
        selected = mask == identifier
        tint[selected] = color
        scored |= selected
    blended = cv2.addWeighted(rgb, 0.48, tint, 0.52, 0.0)
    result[scored] = blended[scored]
    return result


def _write_overlays(overlay_dir, exception, record, rgb, mask):
    seed = int(exception["seed"])
    overlay = _overlay_image(rgb, mask)
    vial_bbox = record.get("label_stats", {}).get("vial_bbox_xyxy")
    if not isinstance(vial_bbox, list) or len(vial_bbox) != 4:
        raise ValueError(f"seed {seed} has no valid vial bounding box")
    x0, y0, x1, y1 = (int(value) for value in vial_bbox)
    padding = 16
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(rgb.shape[1] - 1, x1 + padding)
    y1 = min(rgb.shape[0] - 1, y1 + padding)
    crop = overlay[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        raise ValueError(f"seed {seed} produced an empty overlay crop")

    files = []
    for kind, image in (("full", overlay), ("vial-crop", crop)):
        filename = f"seed-{seed}-{kind}-mask-on-rgb.png"
        path = Path(overlay_dir) / filename
        if path.exists():
            raise FileExistsError(f"refusing to overwrite overlay: {path}")
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write overlay: {path}")
        files.append(
            {
                "filename": filename,
                "relative_ignored_path": _ignored_relative_path(path),
                "sha256": file_sha256(path),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
            }
        )
    return files


def _ignored_relative_path(path):
    path = Path(path)
    if "outputs" in path.parts:
        index = path.parts.index("outputs")
        return Path(*path.parts[index:]).as_posix()
    return Path(path.parent.name, path.name).as_posix()


def review_exceptions(summary_path, collected_root, overlay_dir):
    summary_path = Path(summary_path)
    collected_root = Path(collected_root)
    overlay_dir = Path(overlay_dir)
    summary = _read_json(summary_path, "batch summary")
    if not summary.get("integrity", {}).get("passed"):
        raise ValueError("batch summary integrity is not passed")
    review = summary.get("old_rule_review")
    if not isinstance(review, dict):
        raise ValueError("batch summary has no old_rule_review object")
    exceptions = review.get("exceptions")
    if not isinstance(exceptions, list):
        raise ValueError("old_rule_review.exceptions must be a list")
    _require_equal(len(exceptions), review.get("exception_count"), "exception count")
    expected_role_counts = summary.get("batch_contract", {}).get(
        "expected_role_counts"
    )
    if not isinstance(expected_role_counts, dict):
        raise ValueError("batch summary has no expected role counts")
    for role in ("train", "val"):
        if not isinstance(expected_role_counts.get(role), int):
            raise ValueError(f"batch summary has no integer expected count for {role}")
    if overlay_dir.exists():
        raise FileExistsError(f"refusing to overwrite overlay directory: {overlay_dir}")
    overlay_dir.mkdir(parents=True)

    reviewed = []
    excluded = []
    source_hashes_before = {}
    for exception in exceptions:
        if not isinstance(exception, dict):
            raise ValueError("each exception must be a JSON object")
        chunk = str(exception.get("chunk", ""))
        attempt = _complete_attempt(collected_root / chunk)
        output_dir = attempt / "output"
        manifest_path = output_dir / "manifest.jsonl"
        record = _manifest_record(manifest_path, exception.get("seed"))
        rgb_path, mask_path, rgb, mask, measured = _verify_record(
            exception, record, output_dir
        )
        source_hashes_before[rgb_path] = file_sha256(rgb_path)
        source_hashes_before[mask_path] = file_sha256(mask_path)
        overlays = _write_overlays(overlay_dir, exception, record, rgb, mask)
        disposition = _approved_disposition(exception, measured)
        if disposition["disposition"] == "excluded" and exception.get("role") in {
            "train",
            "val",
        }:
            excluded.append(int(exception["seed"]))
        reviewed.append(
            {
                "seed": int(exception["seed"]),
                "role": exception["role"],
                "chunk": chunk,
                "attempt": attempt.name,
                "frame_index": int(exception["frame_index"]),
                "level_ml": float(exception["level_ml"]),
                "sampling_band": exception["sampling_band"],
                "failed_old_rules": exception["failed_old_rules"],
                "class_pixels": measured,
                "foam": exception["foam"],
                "sticker": exception["sticker"],
                "holder_projection": exception.get("holder_projection"),
                "source": {
                    "manifest_sha256": file_sha256(manifest_path),
                    "rgb_relative_path": record["outputs"]["rgb"]["path"],
                    "rgb_sha256": record["outputs"]["rgb"]["sha256"],
                    "mask_relative_path": record["outputs"]["mask"]["path"],
                    "mask_sha256": record["outputs"]["mask"]["sha256"],
                },
                "overlays": overlays,
                "overlay_relative_ignored_path": overlays[0][
                    "relative_ignored_path"
                ],
                "overlay_sha256": overlays[0]["sha256"],
                **disposition,
            }
        )

    for path, expected_hash in source_hashes_before.items():
        _require_equal(file_sha256(path), expected_hash, f"source immutability for {path.name}")
    excluded_by_role = {
        role: sum(
            item["disposition"] == "excluded" and item["role"] == role
            for item in reviewed
        )
        for role in ("train", "val")
    }
    return {
        "schema_version": 1,
        "purpose": "Step 7 pre-training review of retired visibility-rule exceptions",
        "evidence_label": "measured_from_copied_synthetic_batch",
        "batch_summary_sha256": file_sha256(summary_path),
        "status": "complete",
        "exception_count": len(exceptions),
        "reviewed_count": len(reviewed),
        "unresolved_count": 0,
        "excluded_train_validation_seeds": sorted(set(excluded)),
        "eligible_counts": {
            role: int(expected_role_counts[role]) - excluded_by_role[role]
            for role in ("train", "val")
        },
        "overlay_directory": "ignored Step 7 exception-review output",
        "exceptions": reviewed,
        "policy": (
            "Training may load train/validation data only when status is complete. "
            "Every unexplained train/validation exception is dispositioned as excluded."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--collected-root", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.report_out.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.report_out}")
    report = review_exceptions(args.summary, args.collected_root, args.overlay_dir)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "exceptions": report["exception_count"],
                "excluded": report["excluded_train_validation_seeds"],
                "report": str(args.report_out.resolve()),
            }
        )
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

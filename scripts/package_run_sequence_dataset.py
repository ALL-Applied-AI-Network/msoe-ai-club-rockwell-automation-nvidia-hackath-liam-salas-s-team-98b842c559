#!/usr/bin/env python3
"""Package a complete accepted 30-vial cohort into run/video handoff form."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import cv2


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compose_run_sequence_videos import compose_case_video  # noqa: E402
from fluidmix.physx_varied_cohort_manifest_v3 import (  # noqa: E402
    build_varied_cohort_artifacts,
)
from fluidmix.run_sequence_dataset import (  # noqa: E402
    INDEX_FIELDS,
    build_dataset_index_rows,
    build_model_input_rows,
    validate_case_artifacts,
    validate_dataset_index,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _write_csv_new(
    path: Path, rows: list[dict[str, object]], *, fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fieldnames})


def _probe_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"video is not decodable: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = 0
    width = None
    height = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"video has a non-RGB frame: {path}")
            frame_height, frame_width = frame.shape[:2]
            if width is None:
                width, height = frame_width, frame_height
            elif (frame_width, frame_height) != (width, height):
                raise ValueError(f"video changes resolution: {path}")
            count += 1
    finally:
        capture.release()
    if count == 0 or fps <= 0.0 or width is None or height is None:
        raise ValueError(f"video has invalid timing or frames: {path}")
    return {
        "frame_count": count,
        "fps": fps,
        "resolution": [width, height],
        "duration_s": count / fps,
        "sha256": _sha256(path),
    }


def stitch_run_video(input_paths: list[Path], output_path: Path) -> dict[str, object]:
    """Concatenate six validated vial videos into one fixed-FPS run video."""

    if len(input_paths) != 6:
        raise ValueError("a run video requires exactly six vial clips")
    if output_path.exists():
        raise ValueError(f"run video already exists: {output_path}")
    probes = [_probe_video(path) for path in input_paths]
    fps = float(probes[0]["fps"])
    resolution = list(probes[0]["resolution"])
    if any(abs(float(row["fps"]) - fps) > 1.0e-6 for row in probes):
        raise ValueError("vial clips in one run have different FPS")
    if any(row["resolution"] != resolution for row in probes):
        raise ValueError("vial clips in one run have different resolutions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial.mp4")
    writer = cv2.VideoWriter(
        str(partial),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(resolution[0]), int(resolution[1])),
    )
    if not writer.isOpened():
        raise RuntimeError("run MP4 writer did not open")
    written = 0
    try:
        for path in input_paths:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"vial clip became undecodable: {path}")
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    writer.write(frame)
                    written += 1
            finally:
                capture.release()
    finally:
        writer.release()
    partial.replace(output_path)
    output_probe = _probe_video(output_path)
    expected = sum(int(row["frame_count"]) for row in probes)
    if written != expected or output_probe["frame_count"] != expected:
        raise RuntimeError("stitched run video frame count differs from its inputs")
    return {
        "schema_version": 1,
        "artifact_type": "run_sequence_stitched_run_video_v1",
        "status": "pass",
        "input_videos": [str(path) for path in input_paths],
        "input_video_sha256": [str(row["sha256"]) for row in probes],
        "input_frame_counts": [int(row["frame_count"]) for row in probes],
        "output_video": str(output_path),
        "output_video_sha256": output_probe["sha256"],
        "output_frame_count": output_probe["frame_count"],
        "fps": output_probe["fps"],
        "resolution": output_probe["resolution"],
        "duration_s": output_probe["duration_s"],
        "camera_transform_changed": False,
        "qualified_fill_receipts_changed": False,
    }


def _height_warnings(cohort_root: Path) -> dict[str, list[dict[str, object]]]:
    merge_path = cohort_root / "merge-receipt.json"
    if not merge_path.exists():
        return {}
    merge = _read_object(merge_path, "merge receipt")
    if merge.get("status") != "pass" or merge.get("merged_case_count") != 30:
        raise ValueError("merge receipt is not a complete 30-case pass")
    result: dict[str, list[dict[str, object]]] = {}
    shards = merge.get("source_shards")
    if not isinstance(shards, list):
        raise ValueError("merge receipt lacks source shards")
    for shard in shards:
        warnings_by_case = shard.get("height_warnings_by_case") if isinstance(shard, dict) else None
        if not isinstance(warnings_by_case, dict):
            raise ValueError("merge receipt has invalid height warning provenance")
        for case_id, warnings in warnings_by_case.items():
            if case_id in result or not isinstance(warnings, list):
                raise ValueError("merge receipt height warning mapping is invalid")
            result[str(case_id)] = [dict(row) for row in warnings]
    return result


def _copy_case_evidence(source_case: Path, package_case: Path) -> None:
    app = source_case / "app-output"
    package_case.mkdir(parents=True)
    for directory in ("frames", "liquid", "semantic"):
        source = app / directory
        if not source.is_dir():
            raise ValueError(f"case evidence directory is missing: {source}")
        shutil.copytree(source, package_case / directory, copy_function=shutil.copy2)
    for source, destination in (
        (app / "receipt.json", package_case / "receipt.json"),
        (source_case / "case-result.json", package_case / "case-result.json"),
        (source_case / "launch-receipt.json", package_case / "launch-receipt.json"),
        (source_case / "exit-code.txt", package_case / "exit-code.txt"),
    ):
        if not source.is_file():
            raise ValueError(f"case receipt is missing: {source}")
        shutil.copy2(source, destination)


def _write_readme(
    path: Path,
    *,
    index_validation: dict[str, object],
    height_warning_count: int,
) -> None:
    text = f"""# Synthetic full run-sequence carryover cohort

This package contains {index_validation['case_count']} synthetic vial fills in {index_validation['run_count']} whole production runs. It is a simulation artifact only: residue severity, run colors, and all labels are synthetic controls, not a calibration or validation of a real line.

## Training entry point

Use `model-inputs/INDEX.csv`. Its 30 video paths are opaque case IDs and its fields exclude disposition, prescribed residue, composition, palette, particle state, warnings, and every other scoring oracle. Do not use `videos/INDEX.csv` as a model feature table; it is the audit/handoff index and intentionally contains label and simulator provenance.

Splits are assigned by whole run: 18 train vials (9 KEEP / 9 DISCARD), 6 validation vials (3 / 3), and 6 evaluation vials (3 / 3). Every run contains six vials and is balanced 3 / 3.

## Contents

- `videos/runs/`: five complete run videos in vial order.
- `videos/vials/`: one labeled audit video per vial plus a post-fill transport receipt.
- `videos/INDEX.csv`: run, vial, recipes, prescribed synthetic residue/composition, label, split, palette, height warnings, and evidence paths.
- `model-inputs/videos/` and `model-inputs/INDEX.csv`: label-free training paths and allowlisted input records.
- `cases/`: every rendered RGB still, liquid mask, semantic mask array, renderer receipt, case result, launch receipt, and exit code.
- `manifests/`: source-rebuilt render, model-input, and separated label plans.
- `SHA256SUMS`: hashes for package payload files written before the final package receipt.

## Transport and qualification boundary

The qualified Isaac fill renderer, physics, camera, stream material, mixing, carryover mechanics, validators, and renderer receipt are unchanged. Vial/carrier transport is a deterministic CPU post-fill composition over the accepted final RGB and liquid mask. Its receipt is separate and declares that no camera transform or qualified fill receipt changed.

All renderer receipts passed their hard gates, all receipt-bound stage RGB files are byte-unique within each vial, and every indexed path was checked. There are {height_warning_count} propagated diagnostic particle-height warnings. Such a warning is explicitly not a physics pass and remains excluded from model inputs.

## Synthetic label rule

`KEEP` means the prescribed leading residual is exactly 0 mL. Any positive prescribed leading residual is `DISCARD`. Positive severities are uncalibrated synthetic controls and must not be interpreted as a real discard threshold.
"""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _write_sha256s(package_root: Path) -> tuple[Path, int]:
    output = package_root / "SHA256SUMS"
    paths = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and path != output
        and path.name != "package-receipt.json"
        and ".partial." not in path.name
    )
    with output.open("x", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{_sha256(path)}  {path.relative_to(package_root).as_posix()}\n")
    return output, len(paths)


def package_dataset(
    *, cohort_root: Path, output_root: Path, repo_root: Path = ROOT
) -> dict[str, object]:
    cohort_root = cohort_root.resolve()
    output_root = output_root.resolve()
    if not (cohort_root / "cases").is_dir():
        raise ValueError("cohort root lacks a cases directory")
    if output_root.exists():
        raise ValueError("output root must not already exist")
    output_root.mkdir(parents=True)

    artifacts = build_varied_cohort_artifacts(repo_root=repo_root)
    warnings_by_case = _height_warnings(cohort_root)
    index_rows = build_dataset_index_rows(
        artifacts, height_warnings_by_case=warnings_by_case
    )
    model_rows = build_model_input_rows(artifacts, index_rows=index_rows)
    expected_cases = {row["case_id"] for row in index_rows}
    actual_cases = {
        path.name for path in (cohort_root / "cases").glob("CASE_*") if path.is_dir()
    }
    if actual_cases != expected_cases:
        raise ValueError("cohort case directory set differs from the source-built plan")

    for name, value in artifacts.items():
        _write_json_new(output_root / "manifests" / name, value)

    case_evidence: list[dict[str, object]] = []
    for row in index_rows:
        case_id = row["case_id"]
        source_case = cohort_root / "cases" / case_id
        evidence = validate_case_artifacts(
            source_case, height_warnings=warnings_by_case.get(case_id)
        )
        case_evidence.append(evidence)
        _copy_case_evidence(source_case, output_root / "cases" / case_id)
        vial_output = output_root / row["per_vial_video_path"]
        transport = compose_case_video(
            case_output=source_case / "app-output", output_mp4=vial_output
        )
        probe = _probe_video(vial_output)
        if probe["frame_count"] != transport["output_frame_count"]:
            raise RuntimeError(f"transport video frame verification failed: {case_id}")
        model_video = output_root / f"model-inputs/videos/{case_id}.mp4"
        model_video.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(vial_output, model_video)
        except OSError:
            shutil.copy2(vial_output, model_video)
        if _sha256(model_video) != _sha256(vial_output):
            raise RuntimeError(f"model video copy verification failed: {case_id}")

    rows_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in index_rows:
        rows_by_run[row["run_id"]].append(row)
    run_receipts: list[dict[str, object]] = []
    for run_id, run_rows in rows_by_run.items():
        ordered = sorted(run_rows, key=lambda row: int(row["vial_ordinal"]))
        input_paths = [output_root / row["per_vial_video_path"] for row in ordered]
        output_path = output_root / ordered[0]["run_video_path"]
        receipt = stitch_run_video(input_paths, output_path)
        receipt["run_id"] = run_id
        receipt_path = output_path.with_suffix(".run.json")
        _write_json_new(receipt_path, receipt)
        run_receipts.append(receipt)

    _write_csv_new(
        output_root / "videos" / "INDEX.csv",
        [dict(row) for row in index_rows],
        fieldnames=list(INDEX_FIELDS),
    )
    model_fields = list(model_rows[0])
    if any(list(row) != model_fields for row in model_rows):
        raise ValueError("model input row fields are inconsistent")
    _write_csv_new(
        output_root / "model-inputs" / "INDEX.csv",
        model_rows,
        fieldnames=model_fields,
    )
    index_validation = validate_dataset_index(index_rows, package_root=output_root)
    for row in model_rows:
        if not (output_root / str(row["video_path"])).is_file():
            raise ValueError("model input video path is incomplete")
    _write_readme(
        output_root / "README.md",
        index_validation=index_validation,
        height_warning_count=sum(len(value) for value in warnings_by_case.values()),
    )
    sums_path, hashed_file_count = _write_sha256s(output_root)
    receipt = {
        "schema_version": 1,
        "artifact_type": "synthetic_run_sequence_dataset_package_v1",
        "status": "pass",
        "source_cohort_root": str(cohort_root),
        "output_root": str(output_root),
        "index_validation": index_validation,
        "case_evidence": case_evidence,
        "run_video_receipts": run_receipts,
        "model_input_case_count": len(model_rows),
        "height_warning_count": sum(len(value) for value in warnings_by_case.values()),
        "sha256sums_path": str(sums_path),
        "sha256sums_sha256": _sha256(sums_path),
        "hashed_payload_file_count": hashed_file_count,
        "qualified_fill_receipts_changed": False,
        "model_inputs_exclude_synthetic_oracles": True,
    }
    _write_json_new(output_root / "package-receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = package_dataset(
            cohort_root=args.cohort_root,
            output_root=args.output_root,
            repo_root=args.repo_root,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"package failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

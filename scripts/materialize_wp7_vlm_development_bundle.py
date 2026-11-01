#!/usr/bin/env python3
"""Materialize one complete V4 development cohort without serializing paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.wp7_vlm_decision import build_vial1_bundle  # noqa: E402
from fluidmix.wp7_vlm_development_materializer import (  # noqa: E402
    COMPLETE_RUNS,
    DevelopmentMaterializerError,
    build_deterministic_decision,
    canonical_sha256,
    crop_vial_roi,
    select_complete_cohort,
)

_FRAME = re.compile(r"-F(\d{4})\Z")
_VIAL_ROI = [283, 194, 338, 417]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_bytes())


def _video(case_root: Path, case_id: str, expected_sha256: str) -> Path:
    path = case_root / case_id / "app-output" / "three-pour.avi"
    if path.is_symlink() or not path.is_file() or _sha(path) != expected_sha256:
        raise DevelopmentMaterializerError(f"accepted measured-evidence video missing/drifted: {case_id}")
    return path


def _extract(path: Path, image_ids: list[str], output: Path) -> tuple[dict[str, dict[str, object]], float]:
    reader = cv2.VideoCapture(str(path))
    fps = float(reader.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise DevelopmentMaterializerError("video frame rate is unavailable")
    locators = {}
    for image_id in image_ids:
        match = _FRAME.search(image_id)
        if match is None:
            raise DevelopmentMaterializerError("tracker image ID lacks a frame index")
        reader.set(cv2.CAP_PROP_POS_FRAMES, int(match.group(1)))
        ok, bgr = reader.read()
        if not ok or bgr.shape != (480, 640, 3):
            raise DevelopmentMaterializerError(f"selected frame unavailable: {image_id}")
        crop, crop_binding = crop_vial_roi(bgr, _VIAL_ROI)
        target = output / f"{image_id}.png"
        ok, encoded = cv2.imencode(".png", crop)
        if not ok:
            raise DevelopmentMaterializerError("PNG encoding failed")
        target.write_bytes(encoded.tobytes())
        locators[image_id] = {
            "path": str(target), "sha256": _sha(target), **crop_binding,
        }
    reader.release()
    return locators, fps


def _measurements(path: Path, selected: dict[str, object]) -> tuple[list[dict[str, object]], str]:
    file_sha = _sha(path)
    value = _json(path)
    records = value.get("records") if isinstance(value, dict) else None
    if (
        file_sha != "87578e5cd70bd90284607f8afe2bcb0f661abf6193a58207dfc28406d61a5a8a"
        or not isinstance(records, list) or len(records) != 9
        or value.get("mask_source") != "rgb_only_prediction"
    ):
        raise DevelopmentMaterializerError("accepted train measurement artifact differs")
    rows = sorted(records, key=lambda row: (row.get("vial_ordinal"), row.get("stage_ordinal")))
    if any(
        row.get("run_id") != selected["run_id"]
        or row.get("block_recipe_id") != selected["batch_id"]
        or row.get("limit_codes") != []
        or row.get("mask_source") != "predicted"
        for row in rows
    ):
        raise DevelopmentMaterializerError("accepted train measurements are not cohort-bound")
    return rows, file_sha


def materialize(args: argparse.Namespace) -> dict[str, object]:
    if args.run_id != "RUN-V4-TRAIN-01":
        raise DevelopmentMaterializerError(
            f"{args.run_id} has complete tracker inventory but no accepted nine-record CNN measurement artifact"
        )
    if args.output.exists() or args.output.is_symlink():
        raise DevelopmentMaterializerError("output already exists")
    inventory, plan = _json(args.tracker_inventory), _json(args.plan)
    selected = select_complete_cohort(
        inventory, plan, run_id=args.run_id,
        inventory_file_sha256=_sha(args.tracker_inventory), plan_file_sha256=_sha(args.plan),
    )
    measured, measured_sha = _measurements(args.measurements, selected)
    args.output.mkdir(parents=True)
    image_dir = args.output / "images"
    image_dir.mkdir()
    locators, fps_values = {}, []
    for vial, tracker in zip(selected["vials"], selected["tracker_receipts"], strict=True):
        case_rows = [row for row in measured if row["case_id"] == vial["case_id"]]
        if len(case_rows) != 3 or len({row["clip_sha256"] for row in case_rows}) != 1:
            raise DevelopmentMaterializerError("accepted case measurements are incomplete")
        video = _video(args.case_root, str(vial["case_id"]), str(case_rows[0]["clip_sha256"]))
        image_ids = [stage["images"][phase] for stage in tracker["stages"] for phase in ("early", "mid", "end")]
        bound, fps = _extract(video, image_ids, image_dir)
        locators.update(bound); fps_values.append(fps)
    if len(set(fps_values)) != 1:
        raise DevelopmentMaterializerError("cohort camera frame rates differ")
    stages_by_ordinal = []
    for ordinal in range(1, 4):
        stage_rows = [row for row in measured if row["stage_ordinal"] == ordinal]
        stage_rows.sort(key=lambda row: row["vial_ordinal"])
        stages_by_ordinal.append(stage_rows)
    decision = build_deterministic_decision(
        cohort_id=str(selected["cohort_id"]), target_vial_id=str(selected["vials"][0]["vial_id"]),
        stage_measurements=stages_by_ordinal, source_artifact_sha256=measured_sha,
    )
    target_tracker = selected["tracker_receipts"][0]
    stages = []
    for ordinal, tracker_stage in enumerate(target_tracker["stages"], 1):
        indices = [int(_FRAME.search(tracker_stage["images"][phase]).group(1)) for phase in ("early", "mid", "end")]
        stages.append({
            "stage_id": measured[ordinal - 1]["stage_id"], "stage_ordinal": ordinal,
            "tank_id": tracker_stage["tank_id"], "programmed_ml": tracker_stage["expected_dispense_ml"],
            "dispense_order": ordinal, "stage_timing_seconds": [indices[0] / fps_values[0], indices[-1] / fps_values[0]],
            "stream": {"evidence_id": f"STREAM-{selected['cohort_id']}-S{ordinal}", "median_lab": tracker_stage["stream_lab"], "spread_lab": tracker_stage["stream_spread_lab"]},
        })
    vials = []
    for vial in selected["vials"]:
        vial_rows = [row for row in measured if row["vial_ordinal"] == vial["vial_position"]]
        vial_rows.sort(key=lambda row: row["stage_ordinal"])
        vial_stages = []
        for row, tracker_stage, det_stage in zip(vial_rows, selected["tracker_receipts"][vial["vial_position"] - 1]["stages"], decision["stages"], strict=True):
            measurement = {"evidence_id": row["evidence_id"], "median_lab": row["median_lab"], "mad_lab": row["mad_lab"], "iqr_lab": row["iqr_lab"], "delta_e76_vial2": det_stage["delta_e76_vial2"], "delta_e76_vial3": det_stage["delta_e76_vial3"], "deterministic_disposition": det_stage["disposition"], "limit_codes": det_stage["limit_codes"]}
            if vial["vial_position"] != 1:
                for field in ("delta_e76_vial2", "delta_e76_vial3", "deterministic_disposition"):
                    measurement.pop(field)
            vial_stages.append({"stage_id": row["stage_id"], "frames": [{"image_id": tracker_stage["images"][phase], "phase": phase, "timestamp_seconds": int(_FRAME.search(tracker_stage["images"][phase]).group(1)) / fps_values[0]} for phase in ("early", "mid", "end")], "measurement": measurement})
        vials.append({"vial_position": vial["vial_position"], "vial_id": vial["vial_id"], "stages": vial_stages})
    cohort = {"schema_version": "wp7-vlm-cohort-v1", "cohort_id": selected["cohort_id"], "partition": "twin_train", "recipe_id": selected["recipe_id"], "previous_recipe_id": selected["target_previous_recipe_id"], "camera_metadata": {"resolution": [640, 480], "frame_rate_hz": fps_values[0], "exposure": "unavailable", "white_balance": "unavailable", "viewpoint_id": "synthetic-fillcam-v4", "vial_roi": _VIAL_ROI}, "stages": stages, "vials": vials, "deterministic_decision_receipt": decision}
    cohort_path = args.output / "cohort.json"; cohort_path.write_bytes(json.dumps(cohort, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    built = build_vial1_bundle(cohort, locators, tracker_receipts=selected["tracker_receipts"])
    bundle_path = args.output / "bundle.json"; bundle_path.write_bytes(json.dumps(built.prompt_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    image_inventory = {
        image_id: {key: value for key, value in locator.items() if key != "path"}
        for image_id, locator in sorted(locators.items())
    }
    image_inventory_path = args.output / "image-inventory.json"; image_inventory_path.write_bytes(json.dumps(image_inventory, sort_keys=True, separators=(",", ":")).encode())
    image_locators = {
        image_id: {**locator, "path": f"images/{image_id}.png"}
        for image_id, locator in sorted(locators.items())
    }
    image_locators_path = args.output / "image-locators.json"; image_locators_path.write_bytes(json.dumps(image_locators, sort_keys=True, separators=(",", ":")).encode())
    tracker_dir = args.output / "tracker-receipts"; tracker_dir.mkdir()
    tracker_paths = []
    for position, tracker in enumerate(selected["tracker_receipts"], 1):
        path = tracker_dir / f"vial{position}-tracker-receipt.json"
        path.write_bytes(json.dumps(tracker, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        tracker_paths.append(path)
    reloaded_locators = {image_id: {"path": str(args.output / value["path"]), "sha256": value["sha256"]} for image_id, value in json.loads(image_locators_path.read_bytes()).items()}
    rebuilt = build_vial1_bundle(json.loads(cohort_path.read_bytes()), reloaded_locators, tracker_receipts=[json.loads(path.read_bytes()) for path in tracker_paths])
    if json.dumps(rebuilt.prompt_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() != bundle_path.read_bytes():
        raise DevelopmentMaterializerError("published cohort/locator/receipt rebuild differs")
    v3_inventory = {"schema_version": "wp7-vlm-development-inventory-v1", "dataset_version": "V3", "status": "unavailable", "complete_cohort_count": 0, "reason_code": "no_complete_accepted_three_vial_tracker_cohort", "source_materialization_receipt_sha256": selected["development_binding"]["source_materialization_receipt_sha256"]}
    v3_inventory_path = args.output / "v3-inventory.json"; v3_inventory_path.write_bytes(json.dumps(v3_inventory, sort_keys=True, separators=(",", ":")).encode())
    v3_manifest = {"schema_version": "wp7-vlm-development-input-manifest-v1", "dataset_version": "V3", "partition": "twin_train", "data_origin": "synthetic_twin", "artifacts": [{"artifact_id": "v3-fail-closed-completeness-inventory", "path": "v3-inventory.json", "sha256": _sha(v3_inventory_path)}]}
    v3_manifest_path = args.output / "v3-development-manifest.json"; v3_manifest_path.write_bytes(json.dumps(v3_manifest, sort_keys=True, separators=(",", ":")).encode())
    v4_artifacts = [{"artifact_id": "run-v4-train-01-vlm-cohort", "path": "cohort.json", "sha256": _sha(cohort_path)}, {"artifact_id": "run-v4-train-01-vlm-bundle", "path": "bundle.json", "sha256": _sha(bundle_path)}, {"artifact_id": "run-v4-train-01-image-inventory", "path": "image-inventory.json", "sha256": _sha(image_inventory_path)}, {"artifact_id": "run-v4-train-01-image-locators", "path": "image-locators.json", "sha256": _sha(image_locators_path)}]
    v4_artifacts.extend({"artifact_id": f"run-v4-train-01-vial{position}-tracker-receipt", "path": f"tracker-receipts/{path.name}", "sha256": _sha(path)} for position, path in enumerate(tracker_paths, 1))
    v4_manifest = {"schema_version": "wp7-vlm-development-input-manifest-v1", "dataset_version": "V4", "partition": "twin_train", "data_origin": "synthetic_twin", "artifacts": v4_artifacts}
    v4_manifest_path = args.output / "v4-development-manifest.json"; v4_manifest_path.write_bytes(json.dumps(v4_manifest, sort_keys=True, separators=(",", ":")).encode())
    receipt = {"schema_version": "wp7-vlm-development-bundle-materialization-v1", "status": "complete", "run_id": args.run_id, "cohort_id": selected["cohort_id"], "image_count": len(locators), "cohort_sha256": _sha(cohort_path), "bundle_sha256": _sha(bundle_path), "image_inventory_sha256": _sha(image_inventory_path), "image_locators_sha256": _sha(image_locators_path), "tracker_receipt_file_sha256": [_sha(path) for path in tracker_paths], "development_manifest_sha256": {"V3": _sha(v3_manifest_path), "V4": _sha(v4_manifest_path)}, "bundle_contract_validation": {"status": "pass", "builder": "build_vial1_bundle", "ordered_image_count": len(built.prompt_payload["images"]), "published_inputs_rebuild_equal": True}, "validation_cohort_status": {"RUN-V4-VALIDATION-02": "blocked_missing_accepted_nine_record_cnn_measurements", "RUN-V4-VALIDATION-03": "blocked_missing_accepted_nine_record_cnn_measurements"}, "source_bindings": {**selected["development_binding"], "accepted_measurements_sha256": measured_sha, "deterministic_config_sha256": decision["config_sha256"], "deterministic_route_file_sha256": "6c2c67f6b7363a7bff53b7eebae345acee180bc780fe13740fbbe370f4a37ba2", "tracker_receipt_sha256": [row["receipt_sha256"] for row in selected["tracker_receipts"]], "materializer_cli_source_sha256": _sha(Path(__file__)), "materializer_module_source_sha256": _sha(ROOT / "src" / "fluidmix" / "wp7_vlm_development_materializer.py"), "bundle_builder_source_sha256": _sha(ROOT / "src" / "fluidmix" / "wp7_vlm_decision.py")}, "decision": {"disposition": decision["disposition"], "receipt_sha256": decision["receipt_sha256"]}}
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path = args.output / "materialization-receipt.json"; receipt_path.write_bytes(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=sorted(COMPLETE_RUNS), required=True)
    parser.add_argument("--tracker-inventory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = materialize(args)
    except (DevelopmentMaterializerError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr); return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":"))); return 0


if __name__ == "__main__":
    raise SystemExit(main())

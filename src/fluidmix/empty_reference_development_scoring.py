"""Score already-frozen Approach-A predictions against development labels."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from fluidmix.part_boundary_manifest import (
    authorize_post_inference_label_join,
    seal_saved_inference_artifact,
)


IMAGE_LOCATORS = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "rgb-mask-predictor-01a18128/approach-a-stage2-partition-fix-attempt-01a18128/"
    "development-locators-v1.json"
)
IMAGE_LOCATORS_SHA256 = "409b907e1c9efdff7ded93777042b65ded5d3aad2b67c541e706819856865845"


def _json_numpy_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        default=_json_numpy_scalar,
    ).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _decode_prediction_mask(record: object) -> np.ndarray:
    if not isinstance(record, dict) or record.get("shape_hw") != [480, 640]:
        raise ValueError("prediction mask must declare the FillCam 480x640 shape")
    if record.get("packbits_bitorder") != "little":
        raise ValueError("prediction mask bit order mismatch")
    try:
        packed = base64.b64decode(record["packed_base64"], validate=True)
    except (KeyError, ValueError) as error:
        raise ValueError("prediction mask payload is invalid") from error
    if len(packed) != 480 * 640 // 8:
        raise ValueError("prediction mask packed length mismatch")
    if hashlib.sha256(packed).hexdigest() != record.get("packed_sha256"):
        raise ValueError("prediction mask hash mismatch")
    mask = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8), bitorder="little"
    ).reshape(480, 640).astype(bool)
    if int(np.count_nonzero(mask)) != record.get("pixel_count"):
        raise ValueError("prediction mask pixel count mismatch")
    return mask


def _load_semantic(path: Path, digest: str) -> np.ndarray:
    if path.is_symlink() or _sha256(path) != digest:
        raise ValueError(f"semantic mask hash mismatch: {path}")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (640, 480):
            raise ValueError(f"expected exact 640x480 semantic PNG: {path}")
        image.load()
        return np.asarray(image, dtype=np.uint8) != 0


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    if denominator == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(prediction & target)) / denominator


def score_frozen_development_predictions(
    *,
    prediction_path: str | Path,
    expected_prediction_sha256: str,
    image_locators_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Score a sealed artifact; label files are inaccessible until after the seal."""

    prediction_path = Path(prediction_path)
    locator_path = Path(image_locators_path)
    output_path = Path(output_path)
    sealed = seal_saved_inference_artifact(
        prediction_path,
        expected_sha256=expected_prediction_sha256,
        artifact_type="empty_reference_rgb_frozen_predictions",
    )
    authorize_post_inference_label_join(__name__, sealed)

    # These bytes are deliberately opened only after the exact prediction artifact is sealed.
    predictions = json.loads(sealed.path.read_bytes())
    if (
        predictions.get("schema_version") != 2
        or predictions.get("artifact_type") != sealed.artifact_type
        or predictions.get("phase_separation", {}).get("phase")
        != "rgb_only_prediction_publish"
        or predictions.get("denominators", {}).get("development_stage_rgb_images") != 6
    ):
        raise ValueError("prediction artifact contract mismatch")
    records = predictions.get("development_records")
    if (
        not isinstance(records, list)
        or len(records) != 3
        or len({row.get("case_id") for row in records if isinstance(row, dict)}) != 3
        or len({row.get("vial_id") for row in records if isinstance(row, dict)}) != 3
    ):
        raise ValueError("prediction artifact must contain three unique development cases")
    stage_records = [
        stage
        for row in records
        if isinstance(row, dict) and isinstance(row.get("stages"), list)
        for stage in row["stages"]
    ]
    if (
        len(stage_records) != 6
        or not all(isinstance(stage, dict) for stage in stage_records)
        or len({stage.get("image_id") for stage in stage_records}) != 6
    ):
        raise ValueError("prediction artifact must contain six unique stage predictions")
    if locator_path.resolve() != IMAGE_LOCATORS or _sha256(locator_path) != IMAGE_LOCATORS_SHA256:
        raise ValueError("image locators are not the frozen validation-only artifact")
    locator_artifact = json.loads(locator_path.read_bytes())
    locators = locator_artifact.get("images")
    expected_image_ids = {stage["image_id"] for stage in stage_records}
    if (
        locator_artifact.get("schema_version") != 1
        or locator_artifact.get("artifact_type")
        != "empty_reference_rgb_development_locators"
        or locator_artifact.get("partition") != "validation"
        or not isinstance(locators, dict)
        or set(locators) != expected_image_ids
        or len(locators) != 6
        or any(not isinstance(row, dict) for row in locators.values())
        or any(row.get("split_role") != "validation" for row in locators.values())
    ):
        raise ValueError("expected the exact validation-only development locator artifact")

    rows = []
    for case in records:
        case_root = Path(case["video_path"]).parent
        for stage in case["stages"]:
            image_id = stage["image_id"]
            locator = locators.get(image_id)
            if (
                not isinstance(locator, dict)
                or locator.get("batch_id") != case["batch_id"]
                or locator.get("run_id") != case["run_id"]
                or locator.get("vial_id") != case["vial_id"]
                or locator.get("split_role") != "validation"
                or locator.get("stage_ordinal") != stage["stage_ordinal"]
                or locator.get("tank_id") != stage["tank_id"]
                or locator.get("rgb_path") != stage["rgb_path"]
                or locator.get("rgb_sha256") != stage["rgb_sha256"]
                or not Path(locator.get("mask_path", "")).is_relative_to(case_root)
            ):
                raise ValueError(f"stage locator identity mismatch: {image_id}")
            prediction = _decode_prediction_mask(stage["prediction_mask"])
            target = _load_semantic(Path(locator["mask_path"]), locator["mask_sha256"])
            target_rows = np.flatnonzero(np.any(target, axis=1))
            predicted_rows = np.flatnonzero(np.any(prediction, axis=1))
            target_surface = int(target_rows[0]) if target_rows.size else None
            predicted_surface = int(predicted_rows[0]) if predicted_rows.size else None
            height_error = (
                abs(predicted_surface - target_surface)
                if predicted_surface is not None and target_surface is not None
                else None
            )
            rows.append(
                {
                    "image_id": image_id,
                    "stage_ordinal": stage["stage_ordinal"],
                    "prediction_mask_packed_sha256": stage["prediction_mask"]["packed_sha256"],
                    "semantic_mask_path": locator["mask_path"],
                    "semantic_mask_sha256": locator["mask_sha256"],
                    "dice": _dice(prediction, target),
                    "surface_row_absolute_error_px": height_error,
                }
            )

    dice = [float(row["dice"]) for row in rows]
    height_errors = [
        int(row["surface_row_absolute_error_px"])
        for row in rows
        if row["surface_row_absolute_error_px"] is not None
    ]
    result = {
        "schema_version": 1,
        "artifact_type": "empty_reference_rgb_development_score",
        "status": "development_scored",
        "bindings": {
            "frozen_prediction_path": os.fspath(sealed.path),
            "frozen_prediction_sha256": sealed.sha256,
            "image_locators_path": os.fspath(locator_path),
            "image_locators_sha256": IMAGE_LOCATORS_SHA256,
            "scorer_source_sha256": _sha256(Path(__file__)),
        },
        "phase_separation": {
            "phase": "post_inference_semantic_scoring",
            "prediction_was_hash_sealed_before_locator_or_mask_open": True,
            "prediction_artifact_written_by_scorer": False,
        },
        "denominators": {
            "development_stage_predictions": len(rows),
            "development_semantic_masks_opened": len(rows),
            "held_out_images_opened": 0,
            "held_out_videos_opened": 0,
            "video5_inputs_opened": 0,
        },
        "metrics": {
            "mean_dice": float(np.mean(dice)),
            "minimum_dice": min(dice),
            "maximum_dice": max(dice),
            "mean_surface_row_absolute_error_px": float(np.mean(height_errors)),
            "maximum_surface_row_absolute_error_px": max(height_errors),
        },
        "acceptance": {
            "accuracy_gate_preregistered": False,
            "accepted_for_heldout": False,
            "note": "Metrics are validation diagnostics; scoped review is still required before one heldout pass.",
        },
        "per_image": rows,
    }
    _publish_exclusive(output_path, result)
    if _sha256(prediction_path) != sealed.sha256:
        raise RuntimeError("prediction artifact changed during scoring")
    return result


__all__ = ["score_frozen_development_predictions"]

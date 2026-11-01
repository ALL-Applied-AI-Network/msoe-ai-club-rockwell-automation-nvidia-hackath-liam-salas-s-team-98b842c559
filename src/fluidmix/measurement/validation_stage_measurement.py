"""Run the frozen VA pixel-to-evidence measurement boundary."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re

from fluidmix.measurement import validation_measurement_binding as binding
from twin.render.render_process import (
    atomic_write_json,
    hash_regular_file,
    load_strict_json_snapshot,
)
from fluidmix.measurement.vision_measurement import evaluate_model_plan


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOCATOR_FIELDS = {"rgb_path", "rgb_sha256", "mask_path", "mask_sha256"}
_RECEIPT_FIELDS = {
    "schema_version",
    "artifact_type",
    "status",
    "family_id",
    "partition",
    "data_origin",
    "real_data_used",
    "source_sliced_model_input_sha256",
    "source_cohort_graph_sha256",
    "source_cohort_run_sha256",
    "image_locator_sha256",
    "measurement_source",
    "capture_gate_verification",
    "video_gate_verification",
    "review_video_role",
    "stage_time_codes",
    "render_runtime",
    "denominators",
    "claim_boundary",
}


def _mapping(value: object, role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be an object")
    return value


def _validate_receipt(
    value: object,
    *,
    model_sha256: str,
    locator_sha256: str,
) -> Mapping[str, object]:
    receipt = _mapping(value, "validation binding receipt")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("validation binding receipt fields drifted")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != "validation_image_binding_receipt"
        or receipt.get("status") != "bound"
        or receipt.get("family_id") != "VA"
        or receipt.get("partition") != "val"
        or receipt.get("data_origin") != "synthetic_twin"
        or receipt.get("real_data_used") is not False
        or receipt.get("source_sliced_model_input_sha256") != model_sha256
        or receipt.get("image_locator_sha256") != locator_sha256
        or receipt.get("measurement_source")
        != "hash_bound_capture_rgb_and_liquid_mask_pngs"
        or receipt.get("capture_gate_verification")
        != "full_summary_and_artifact_tree_rehashed"
        or receipt.get("video_gate_verification")
        != "recorded_decode_gate_and_mp4_hash_reverified_without_measurement_decode"
        or receipt.get("review_video_role")
        != "verified_lossy_visualization_not_measurement"
        or receipt.get("stage_time_codes") != [171, 342, 686]
        or receipt.get("denominators")
        != {
            "batches": 8,
            "captures": 24,
            "stage_images": 72,
            "verified_videos_not_used_for_measurement": 24,
        }
    ):
        raise ValueError("validation binding receipt provenance drifted")
    for field in ("source_cohort_graph_sha256", "source_cohort_run_sha256"):
        digest = receipt.get(field)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"validation binding receipt {field} is malformed")
    runtime = _mapping(receipt.get("render_runtime"), "validation render runtime")
    if any(runtime.get(key) != expected for key, expected in binding._EXPECTED_RUNTIME.items()):
        raise ValueError("validation binding receipt runtime drifted")
    isaac_python = runtime.get("isaac_python")
    if not isinstance(isaac_python, str) or not Path(isaac_python).is_absolute():
        raise ValueError("validation binding receipt Isaac Python is malformed")
    claim = receipt.get("claim_boundary")
    if not isinstance(claim, str) or "decoded-MP4 measurement" not in claim:
        raise ValueError("validation binding receipt claim boundary drifted")
    return receipt


def _validate_locator(
    value: object,
    *,
    expected_image_ids: set[str],
) -> Mapping[str, object]:
    locator = _mapping(value, "validation image locator")
    if (
        set(locator) != {"schema_version", "data_origin", "images"}
        or locator.get("schema_version") != 1
        or locator.get("data_origin") != "twin"
    ):
        raise ValueError("validation image locator origin or schema drifted")
    images = _mapping(locator.get("images"), "validation image locators")
    if set(images) != expected_image_ids or len(images) != 72:
        raise ValueError("validation image locator IDs drifted")
    for image_id, raw in images.items():
        row = _mapping(raw, f"validation image locator {image_id}")
        if set(row) != _LOCATOR_FIELDS:
            raise ValueError("validation image locator fields drifted")
        for path_field in ("rgb_path", "mask_path"):
            text = row.get(path_field)
            if not isinstance(text, str) or not text:
                raise ValueError("validation image path is malformed")
            path = Path(text)
            if (
                not path.is_absolute()
                or path.suffix.lower() != ".png"
                or path.resolve(strict=True) != path
                or not path.is_file()
            ):
                raise ValueError("validation measurements require direct PNG inputs")
        for hash_field in ("rgb_sha256", "mask_sha256"):
            digest = row.get(hash_field)
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError("validation image hash is malformed")
    return locator


def _rehash_pixel_inputs(locator: Mapping[str, object]) -> None:
    images = _mapping(locator["images"], "validation image locators")
    for raw in images.values():
        row = _mapping(raw, "validation image locator")
        for role in ("rgb", "mask"):
            snapshot = hash_regular_file(Path(str(row[f"{role}_path"])))
            if snapshot["sha256"] != row[f"{role}_sha256"]:
                raise ValueError(f"validation {role} pixels drifted")


def write_validation_stage_evidence(
    model_plan_path: str | os.PathLike[str],
    locator_path: str | os.PathLike[str],
    binding_receipt_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Publish measured evidence from the completed synthetic VA cohort."""

    model, model_snapshot = load_strict_json_snapshot(model_plan_path)
    images = binding._validate_model_slice(model, model_snapshot)
    expected_image_ids = {str(row["image_id"]) for row in images}
    locator, locator_snapshot = load_strict_json_snapshot(locator_path)
    checked_locator = _validate_locator(
        locator, expected_image_ids=expected_image_ids
    )
    receipt, receipt_snapshot = load_strict_json_snapshot(binding_receipt_path)
    _validate_receipt(
        receipt,
        model_sha256=str(model_snapshot["sha256"]),
        locator_sha256=str(locator_snapshot["sha256"]),
    )

    measured = evaluate_model_plan(model, checked_locator)
    denominators = measured.get("denominators")
    if not isinstance(denominators, Mapping) or (
        denominators.get("image_measurements") != 72
        or denominators.get("vial_decisions") != 12
        or denominators.get("valid_vial_decisions", 0)
        + denominators.get("abstained_vial_decisions", 0)
        != 12
    ):
        raise ValueError("validation measurement denominators drifted")
    image_measurements = measured.get("image_measurements")
    if (
        not isinstance(image_measurements, list)
        or len(image_measurements) != 72
        or any(row.get("validity") != "valid" for row in image_measurements)
    ):
        raise ValueError("a validation RGB/mask measurement failed closed")

    measured["source_model_input_sha256"] = model_snapshot["sha256"]
    measured["source_image_locators_sha256"] = locator_snapshot["sha256"]
    measured["source_binding_receipt_sha256"] = receipt_snapshot["sha256"]
    _rehash_pixel_inputs(checked_locator)
    for path, snapshot, role in (
        (model_plan_path, model_snapshot, "model input"),
        (locator_path, locator_snapshot, "image locator"),
        (binding_receipt_path, receipt_snapshot, "binding receipt"),
    ):
        if hash_regular_file(Path(path)) != snapshot:
            raise RuntimeError(f"validation {role} changed during measurement")

    output = Path(output_path)
    atomic_write_json(output.parent, output.name, measured)
    return measured

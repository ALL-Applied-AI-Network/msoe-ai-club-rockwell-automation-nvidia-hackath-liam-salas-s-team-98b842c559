"""Label-free colour measurements for published F2 Varied V3 stage captures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from io import BytesIO
import hashlib
import itertools
import json
import math
from pathlib import Path

from fluidmix.capture_manifest_adapter import (
    CaptureManifestError,
    bind_stage_capture_manifest,
)
from fluidmix.rgb_mask_prediction import validate_rgb_mask_prediction_locators
from fluidmix.render_process import load_strict_json_snapshot
from fluidmix.vision_measurement import _publish_json_no_clobber


class VariedStageMeasurementError(ValueError):
    """A Varied V3 measurement input failed closed."""


def _bytes(path_text: object, expected_sha256: object, role: str) -> bytes:
    if not isinstance(path_text, str) or not path_text:
        raise VariedStageMeasurementError(f"{role}_path_invalid")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise VariedStageMeasurementError(f"{role}_sha256_invalid")
    path = Path(path_text)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise VariedStageMeasurementError(f"{role}_missing: {path}") from exc
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise VariedStageMeasurementError(f"{role}_hash_mismatch: {path}")
    return data


def _robust_pixel_stats(locator: object) -> dict[str, object]:
    if not isinstance(locator, Mapping):
        raise VariedStageMeasurementError("image_locator_malformed")
    rgb_bytes = _bytes(locator.get("rgb_path"), locator.get("rgb_sha256"), "rgb")
    mask_bytes = _bytes(locator.get("mask_path"), locator.get("mask_sha256"), "mask")

    from PIL import Image, UnidentifiedImageError
    import numpy as np
    from skimage.color import rgb2lab

    try:
        with Image.open(BytesIO(rgb_bytes)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise VariedStageMeasurementError("rgb_not_uint8_rgb_png")
            image.load()
            rgb = np.asarray(image, dtype=np.uint8).copy()
        with Image.open(BytesIO(mask_bytes)) as image:
            if image.format != "PNG" or image.mode not in {"L", "1"}:
                raise VariedStageMeasurementError("mask_not_binary_png")
            image.load()
            mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        if isinstance(exc, VariedStageMeasurementError):
            raise
        raise VariedStageMeasurementError("png_decode_failed") from exc
    if rgb.shape != (480, 640, 3) or mask.shape != rgb.shape[:2]:
        raise VariedStageMeasurementError("rgb_mask_shape_mismatch")
    if not set(int(value) for value in np.unique(mask)).issubset({0, 255}):
        raise VariedStageMeasurementError("mask_not_binary")
    selected = mask == 255
    liquid_pixel_count = int(np.count_nonzero(selected))
    if liquid_pixel_count <= 0:
        raise VariedStageMeasurementError("mask_empty")
    lab_pixels = rgb2lab(rgb.astype(np.float64) / 255.0)[selected]
    rgb_pixels = rgb[selected].astype(np.float64)
    median_lab = np.median(lab_pixels, axis=0)
    mad_lab = np.median(np.abs(lab_pixels - median_lab), axis=0)
    q25_lab, q75_lab = np.percentile(lab_pixels, [25.0, 75.0], axis=0)
    median_rgb = np.median(rgb_pixels, axis=0)
    mad_rgb = np.median(np.abs(rgb_pixels - median_rgb), axis=0)
    numeric = np.concatenate(
        (median_lab, mad_lab, q25_lab, q75_lab, median_rgb, mad_rgb)
    )
    if not np.all(np.isfinite(numeric)):
        raise VariedStageMeasurementError("colour_statistic_nonfinite")
    occupied_rows = np.flatnonzero(np.any(selected, axis=1))
    coverage = liquid_pixel_count / float(mask.size)
    if not math.isfinite(coverage) or coverage <= 0.0 or coverage > 1.0:
        raise VariedStageMeasurementError("mask_coverage_invalid")
    return {
        "liquid_pixel_count": liquid_pixel_count,
        "mask_coverage_fraction": coverage,
        "mask_height_px": int(occupied_rows[-1] - occupied_rows[0] + 1),
        "median_lab": median_lab.tolist(),
        "mad_lab": mad_lab.tolist(),
        "iqr_lab": (q75_lab - q25_lab).tolist(),
        "median_srgb_0_255": median_rgb.tolist(),
        "mad_srgb_0_255": mad_rgb.tolist(),
    }


def _delta_e76(left: Sequence[object], right: Sequence[object]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise VariedStageMeasurementError("median_lab_malformed")
    value = math.sqrt(
        sum(
            (float(a) - float(b)) ** 2
            for a, b in zip(left, right, strict=True)
        )
    )
    if not math.isfinite(value):
        raise VariedStageMeasurementError("delta_e76_nonfinite")
    return value


def build_varied_stage_measurements(
    manifests: object,
    *,
    predicted_mask_locators: object | None = None,
) -> dict[str, object]:
    """Build measurements only; decisions and synthetic labels stay downstream."""

    if (
        not isinstance(manifests, (list, tuple))
        or not manifests
        or any(not isinstance(path, (str, Path)) for path in manifests)
    ):
        raise VariedStageMeasurementError("manifest_paths_invalid")

    prediction_images: Mapping[str, object] | None = None
    prediction_producer: dict[str, object] | None = None
    if predicted_mask_locators is not None:
        try:
            prediction_images, prediction_producer = (
                validate_rgb_mask_prediction_locators(predicted_mask_locators)
            )
        except (TypeError, ValueError) as exc:
            raise VariedStageMeasurementError(
                f"prediction_locator_invalid: {exc}"
            ) from exc

    captures: list[dict[str, object]] = []
    seen_paths: set[Path] = set()
    for raw_path in manifests:
        try:
            path = Path(raw_path).resolve(strict=True)
        except OSError as exc:
            raise VariedStageMeasurementError(f"manifest_missing: {raw_path}") from exc
        if path in seen_paths:
            raise VariedStageMeasurementError(f"duplicate_manifest_path: {path}")
        seen_paths.add(path)
        try:
            bound = bind_stage_capture_manifest(path)
        except CaptureManifestError as exc:
            raise VariedStageMeasurementError(str(exc)) from exc
        if bound.get("capture_contract_version") != "f2-brev-isaac62-varied-event-v3":
            raise VariedStageMeasurementError("capture_contract_version_invalid")
        captures.append(bound)

    captures.sort(
        key=lambda row: (
            str(row["split_role"]), str(row["run_id"]), str(row["batch_id"])
        )
    )
    batch_ids = [str(row["batch_id"]) for row in captures]
    run_ids = [str(row["run_id"]) for row in captures]
    if len(set(batch_ids)) != len(batch_ids):
        raise VariedStageMeasurementError("duplicate_batch_id")
    if len(set(run_ids)) != len(run_ids):
        raise VariedStageMeasurementError("duplicate_run_id")

    source_images: dict[str, Mapping[str, object]] = {}
    for capture in captures:
        container = capture.get("image_locators")
        images = container.get("images") if isinstance(container, Mapping) else None
        if not isinstance(images, Mapping):
            raise VariedStageMeasurementError("bound_capture_malformed")
        for image_id, raw_locator in images.items():
            if not isinstance(image_id, str) or not isinstance(raw_locator, Mapping):
                raise VariedStageMeasurementError("bound_capture_malformed")
            if image_id in source_images:
                raise VariedStageMeasurementError("duplicate_image_id")
            source_images[image_id] = raw_locator
    if prediction_images is not None:
        if set(prediction_images) != set(source_images):
            raise VariedStageMeasurementError("prediction_image_set_mismatch")
        predicted_mask_paths: list[str] = []
        for image_id, raw_prediction in prediction_images.items():
            if not isinstance(raw_prediction, Mapping):
                raise VariedStageMeasurementError("prediction_locator_invalid")
            source = source_images[image_id]
            if any(
                raw_prediction.get(field) != source.get(field)
                for field in ("rgb_path", "rgb_sha256")
            ):
                raise VariedStageMeasurementError(
                    f"prediction_rgb_binding_mismatch: {image_id}"
                )
            predicted_mask_paths.append(str(raw_prediction["mask_path"]))
        if len(set(predicted_mask_paths)) != len(predicted_mask_paths):
            raise VariedStageMeasurementError("prediction_mask_path_reused")

    stage_measurements: list[dict[str, object]] = []
    source_manifests: list[dict[str, object]] = []
    stage_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    vial_keys: set[tuple[str, str]] = set()
    two_stage_runs = 0
    three_stage_runs = 0
    for capture in captures:
        events = capture.get("stage_events")
        measurements = capture.get("image_measurements")
        locators_container = capture.get("image_locators")
        if (
            not isinstance(events, list)
            or len(events) not in {2, 3}
            or not isinstance(measurements, list)
            or not isinstance(locators_container, Mapping)
            or not isinstance(locators_container.get("images"), Mapping)
        ):
            raise VariedStageMeasurementError("bound_capture_malformed")
        if len(events) == 2:
            two_stage_runs += 1
        else:
            three_stage_runs += 1
        images = locators_container["images"]
        assert isinstance(images, Mapping)
        expected_stage_set = {
            (str(event["event_id"]), int(event["stage_ordinal"]), str(event["tank_id"]))
            for event in events
        }
        observed_by_vial: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
        for measured in measurements:
            if not isinstance(measured, Mapping) or measured.get("validity") != "valid":
                raise VariedStageMeasurementError("stage_measurement_invalid")
            image_id = measured.get("image_id")
            stage_event = measured.get("stage_event")
            if not isinstance(image_id, str) or not isinstance(stage_event, Mapping):
                raise VariedStageMeasurementError("stage_measurement_malformed")
            vial_id = str(measured.get("vial_id"))
            event_key = (
                str(stage_event.get("event_id")),
                int(stage_event.get("stage_ordinal", 0)),
                str(stage_event.get("tank_id")),
            )
            observed_by_vial[vial_id].add(event_key)
            source_locator = images.get(image_id)
            locator = (
                prediction_images.get(image_id)
                if prediction_images is not None
                else source_locator
            )
            pixel_stats = _robust_pixel_stats(locator)
            existing_lab = measured.get("lab_raw")
            if prediction_images is None and (
                not isinstance(existing_lab, list)
                or len(existing_lab) != 3
                or any(
                    abs(float(left) - float(right)) > 1e-12
                    for left, right in zip(
                        existing_lab, pixel_stats["median_lab"], strict=True
                    )
                )
            ):
                raise VariedStageMeasurementError("median_lab_binding_mismatch")
            assert isinstance(locator, Mapping)
            batch_id = str(capture["batch_id"])
            run_id = str(capture["run_id"])
            record = {
                "evidence_id": f"EVID-{image_id}",
                "measurement_id": f"MEAS-{image_id}",
                "image_id": image_id,
                "batch_id": batch_id,
                "case_key": f"{batch_id}:{vial_id}",
                "run_id": run_id,
                "recipe_id": str(capture["recipe_id"]),
                "split_role": str(capture["split_role"]),
                "split_version": str(capture["split_version"]),
                "vial_id": vial_id,
                "stage_id": str(stage_event["event_id"]),
                "stage_ordinal": int(stage_event["stage_ordinal"]),
                "tank_id": str(stage_event["tank_id"]),
                "view_id": str(capture["view_id"]),
                "source_manifest_sha256": str(capture["source_manifest_sha256"]),
                "rgb_sha256": str(locator["rgb_sha256"]),
                "mask_sha256": str(locator["mask_sha256"]),
                **pixel_stats,
                "lab_unit": "cie_lab_d65_2deg",
            }
            stage_measurements.append(record)
            stage_groups[(batch_id, str(stage_event["event_id"]))].append(record)
            vial_keys.add((batch_id, vial_id))
        if set(observed_by_vial) != {"V1", "V2", "V3"} or any(
            stage_set != expected_stage_set for stage_set in observed_by_vial.values()
        ):
            raise VariedStageMeasurementError("stage_set_mismatch")
        source_manifests.append(
            {
                "batch_id": str(capture["batch_id"]),
                "run_id": str(capture["run_id"]),
                "recipe_id": str(capture["recipe_id"]),
                "split_role": str(capture["split_role"]),
                "sha256": str(capture["source_manifest_sha256"]),
                "active_stage_count": len(events),
                "stage_measurement_count": len(measurements),
            }
        )

    stage_measurements.sort(
        key=lambda row: (
            str(row["split_role"]), str(row["run_id"]), str(row["batch_id"]),
            str(row["vial_id"]), int(row["stage_ordinal"]),
        )
    )
    pairs: list[dict[str, object]] = []
    for (batch_id, stage_id), group in sorted(stage_groups.items()):
        ordered = sorted(group, key=lambda row: str(row["vial_id"]))
        if [row["vial_id"] for row in ordered] != ["V1", "V2", "V3"]:
            raise VariedStageMeasurementError("stage_set_mismatch")
        context_fields = (
            "run_id", "recipe_id", "split_role", "split_version", "stage_id",
            "stage_ordinal", "tank_id", "source_manifest_sha256",
        )
        if any(
            any(row[field] != ordered[0][field] for field in context_fields)
            for row in ordered[1:]
        ):
            raise VariedStageMeasurementError("stage_context_mismatch")
        for left, right in itertools.combinations(ordered, 2):
            delta = _delta_e76(left["median_lab"], right["median_lab"])
            pairs.append(
                {
                    "pair_id": (
                        f"PAIR-{batch_id}-{stage_id}-{left['vial_id']}-{right['vial_id']}"
                    ),
                    "batch_id": batch_id,
                    "run_id": left["run_id"],
                    "recipe_id": left["recipe_id"],
                    "split_role": left["split_role"],
                    "split_version": left["split_version"],
                    "stage_id": stage_id,
                    "stage_ordinal": left["stage_ordinal"],
                    "tank_id": left["tank_id"],
                    "left_vial_id": left["vial_id"],
                    "right_vial_id": right["vial_id"],
                    "left_evidence_id": left["evidence_id"],
                    "right_evidence_id": right["evidence_id"],
                    "source_manifest_sha256": left["source_manifest_sha256"],
                    "left_rgb_sha256": left["rgb_sha256"],
                    "right_rgb_sha256": right["rgb_sha256"],
                    "left_mask_sha256": left["mask_sha256"],
                    "right_mask_sha256": right["mask_sha256"],
                    "delta_e76": delta,
                    "delta_e76_6dp": f"{delta:.6f}",
                    "unit": "delta_e76",
                }
            )

    result = {
        "schema_version": 1,
        "artifact_type": "f2_varied_stage_measured_evidence",
        "status": "measured",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
        "source_manifests": source_manifests,
        "stage_measurements": stage_measurements,
        "same_run_stage_pairs": pairs,
        "measurement_contract": {
            "pixels": "sha256_bound_png_rgb_under_aligned_binary_liquid_mask",
            "case_identity": "batch_id_plus_vial_id",
            "location": "coordinatewise_median_cie_lab_d65_2deg",
            "dispersion": "coordinatewise_lab_mad_and_iqr",
            "distance": "cie76_between_coordinatewise_median_lab",
            "scope": "measurements_only_no_keep_discard_or_abstain_output",
        },
        "denominators": {
            "source_manifests": len(captures),
            "runs": len(run_ids),
            "vials": len(vial_keys),
            "active_stage_events": len(stage_groups),
            "stage_measurements": len(stage_measurements),
            "same_run_stage_pairs": len(pairs),
            "two_stage_runs": two_stage_runs,
            "three_stage_runs": three_stage_runs,
        },
        "claim_boundary": (
            "Synthetic-twin masked colour measurements only; no keep/discard inference, "
            "physical-colour validation, or sim-to-real claim."
        ),
    }
    if prediction_producer is not None:
        result["mask_source"] = "rgb_only_prediction"
        result["mask_prediction_producer"] = prediction_producer
        result["measurement_contract"]["pixels"] = (
            "sha256_bound_png_rgb_under_rgb_only_predicted_binary_liquid_mask"
        )
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VariedStageMeasurementError("output_nonfinite_or_unserializable") from exc
    return result


def write_varied_stage_measurements(
    manifests: object,
    output_path: str | Path,
    *,
    predicted_image_locators_path: str | Path | None = None,
) -> dict[str, object]:
    """Build and publish one immutable JSON artifact without clobbering."""

    output = Path(output_path)
    predictions: object | None = None
    prediction_sha256: str | None = None
    if predicted_image_locators_path is not None:
        try:
            predictions, snapshot = load_strict_json_snapshot(
                predicted_image_locators_path
            )
        except ValueError as exc:
            raise VariedStageMeasurementError(
                f"prediction_locator_unavailable: {predicted_image_locators_path}"
            ) from exc
        prediction_sha256 = str(snapshot["sha256"])
    result = build_varied_stage_measurements(
        manifests,
        predicted_mask_locators=predictions,
    )
    if prediction_sha256 is not None:
        result["source_prediction_locator_sha256"] = prediction_sha256
    _publish_json_no_clobber(output, result)
    return result

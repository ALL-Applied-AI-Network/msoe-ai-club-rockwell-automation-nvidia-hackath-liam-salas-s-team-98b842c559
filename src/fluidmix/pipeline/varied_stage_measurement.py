"""Bind published capture paths before invoking pure varied-stage measurement."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fluidmix.core.io import load_strict_json_snapshot
from fluidmix.measurement.varied_stage_measurement import (
    VariedStageMeasurementError,
    build_varied_stage_measurements as _build_varied_stage_measurements,
)
from fluidmix.measurement.vision_measurement import _publish_json_no_clobber
from fluidmix.perception.masks import (
    validate_rgb_mask_prediction_locators,
    validate_temporal_rgb_mask_prediction_locators,
)
from fluidmix.storage.artifacts.capture_manifest_adapter import (
    CaptureManifestError,
    bind_predicted_stage_capture_manifest,
    bind_stage_capture_manifest,
)


def build_varied_stage_measurements(
    manifests: object,
    *,
    predicted_mask_locators: object | None = None,
) -> dict[str, object]:
    """Bind capture manifests, then pass their image values into measurement."""

    if (
        not isinstance(manifests, (list, tuple))
        or not manifests
        or any(not isinstance(path, (str, Path)) for path in manifests)
    ):
        raise VariedStageMeasurementError("manifest_paths_invalid")
    prediction_values: dict[str, object] | None = None
    if predicted_mask_locators is not None:
        try:
            temporal = (
                isinstance(predicted_mask_locators, Mapping)
                and predicted_mask_locators.get("schema_version") == 3
            )
            if temporal:
                prediction_images, prediction_producer = (
                    validate_temporal_rgb_mask_prediction_locators(
                        predicted_mask_locators
                    )
                )
            else:
                prediction_images, prediction_producer = (
                    validate_rgb_mask_prediction_locators(predicted_mask_locators)
                )
        except (TypeError, ValueError) as exc:
            raise VariedStageMeasurementError(
                f"prediction_locator_invalid: {exc}"
            ) from exc
        prediction_values = {
            "images": prediction_images,
            "producer": prediction_producer,
            "temporal": temporal,
        }
    captures: list[Mapping[str, object]] = []
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
            bound = (
                bind_predicted_stage_capture_manifest(path)
                if predicted_mask_locators is not None
                else bind_stage_capture_manifest(path)
            )
        except CaptureManifestError as exc:
            raise VariedStageMeasurementError(str(exc)) from exc
        captures.append(bound)
    return _build_varied_stage_measurements(
        captures,
        prediction_values=prediction_values,
    )


def write_varied_stage_measurements(
    manifests: object,
    output_path: str | Path,
    *,
    predicted_image_locators_path: str | Path | None = None,
    bridge_compatible: bool = False,
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
        if bridge_compatible:
            contract = result.get("measurement_contract")
            if not isinstance(contract, dict):
                raise VariedStageMeasurementError("measurement_contract_malformed")
            contract["source_prediction_locator_sha256"] = prediction_sha256
        else:
            result["source_prediction_locator_sha256"] = prediction_sha256
    _publish_json_no_clobber(output, result)
    return result


__all__ = [
    "VariedStageMeasurementError",
    "build_varied_stage_measurements",
    "write_varied_stage_measurements",
]

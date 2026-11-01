"""Adapt accepted camera fill-step receipts to phase endpoint plans.

This consumer intentionally owns no stream, colour, CNN, or volume thresholds.
The tracker producer validates its closed receipt; this module only binds that
accepted result to the claimed recipe and trusted image/frame registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import re


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class TrackerReceiptAdapterError(ValueError):
    """An accepted tracker receipt could not be bound to the phase inputs."""


@dataclass(frozen=True)
class TrackerEndpointStage:
    """One accepted tracker stage resolved to trusted phase-frame identities."""

    fill_step: int
    tank_id: str
    expected_dispense_ml: float
    expected_cumulative_ml: float
    twin_geometry_equivalent_volume_ml: float
    agreement: bool
    stream_lab: tuple[float, float, float]
    stream_spread_lab: tuple[float, float, float]
    early_image_id: str
    mid_image_id: str
    end_image_id: str
    early_frame_index: int
    mid_frame_index: int
    end_frame_index: int


@dataclass(frozen=True)
class TrackerEndpointPlan:
    """Hash-bound stage endpoints for one recipe-sequence case."""

    case_id: str
    vial_id: str
    cohort_id: str
    receipt_sha256: str
    source_artifact_sha256: str
    calibration_sha256: str
    recipe_artifact_sha256: str
    stages: tuple[TrackerEndpointStage, ...]


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrackerReceiptAdapterError(f"{role} must be a lowercase SHA-256")
    return value


def _number(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrackerReceiptAdapterError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TrackerReceiptAdapterError(f"{role} must be finite and positive")
    return result


def _triplet(value: object, role: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise TrackerReceiptAdapterError(f"{role} must contain three channels")
    converted = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TrackerReceiptAdapterError(f"{role} must contain three channels")
        number = float(item)
        if not math.isfinite(number):
            raise TrackerReceiptAdapterError(f"{role} must contain finite channels")
        converted.append(number)
    return converted[0], converted[1], converted[2]


def _producer_validator() -> Callable[[object], Mapping[str, object]]:
    try:
        from ..perception.tracker.fill_step_tracker import validate_fill_step_receipt
    except ImportError as error:
        raise TrackerReceiptAdapterError(
            "tracker producer validator is not available; merge the accepted "
            "wp7_fill_step_tracker contract before production integration"
        ) from error
    return validate_fill_step_receipt


def _recipe_vials(recipe_sequence: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(recipe_sequence, Mapping):
        raise TrackerReceiptAdapterError("recipe sequence must be a mapping")
    runs = recipe_sequence.get("runs")
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence) or not runs:
        raise TrackerReceiptAdapterError("recipe sequence runs are missing")
    result: list[Mapping[str, object]] = []
    case_ids: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            raise TrackerReceiptAdapterError("recipe sequence run is invalid")
        vials = run.get("vials")
        if isinstance(vials, (str, bytes)) or not isinstance(vials, Sequence):
            raise TrackerReceiptAdapterError("recipe sequence vials are invalid")
        for vial in vials:
            if not isinstance(vial, Mapping):
                raise TrackerReceiptAdapterError("recipe sequence vial is invalid")
            case_id = vial.get("case_id")
            if not isinstance(case_id, str) or not case_id or case_id in case_ids:
                raise TrackerReceiptAdapterError("recipe case identity/order is invalid")
            case_ids.add(case_id)
            result.append(vial)
    if not result:
        raise TrackerReceiptAdapterError("recipe sequence has no vials")
    return tuple(result)


def _exact_case_mapping(
    value: object, case_ids: tuple[str, ...], role: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(case_ids):
        raise TrackerReceiptAdapterError(f"{role} must bind every recipe case exactly")
    return value


def _validate_receipt(
    value: object,
    validator: Callable[[object], Mapping[str, object]],
) -> Mapping[str, object]:
    try:
        checked = validator(value)
    except (TypeError, ValueError) as error:
        raise TrackerReceiptAdapterError(
            f"tracker producer validator rejected receipt: {error}"
        ) from error
    if not isinstance(checked, Mapping):
        raise TrackerReceiptAdapterError("tracker producer validator returned no receipt")
    return checked


def adapt_tracker_receipts_to_endpoint_plans(
    *,
    receipts: object,
    recipe_sequence: object,
    receipt_vial_id_by_case: object,
    image_bindings: object,
    expected_cohort_id: str,
    expected_receipt_sha256_by_case: object,
    expected_source_artifact_sha256_by_case: object,
    expected_calibration_sha256: str,
    expected_recipe_artifact_sha256: str,
) -> tuple[TrackerEndpointPlan, ...]:
    """Project accepted fill-step receipts without rerunning tracker policy.

    ``image_bindings`` is the narrow registry projection this seam needs. Each
    opaque image ID binds exactly ``case_id``, ``frame_index``, ``rgb_sha256``
    and ``mask_sha256``. The caller remains responsible for constructing that
    projection from its trusted camera registry.
    """

    vials = _recipe_vials(recipe_sequence)
    case_ids = tuple(str(vial["case_id"]) for vial in vials)
    vial_ids = _exact_case_mapping(
        receipt_vial_id_by_case, case_ids, "tracker vial identity map"
    )
    receipt_hashes = _exact_case_mapping(
        expected_receipt_sha256_by_case, case_ids, "expected tracker receipt hashes"
    )
    source_hashes = _exact_case_mapping(
        expected_source_artifact_sha256_by_case,
        case_ids,
        "expected tracker source hashes",
    )
    if not isinstance(expected_cohort_id, str) or not expected_cohort_id:
        raise TrackerReceiptAdapterError("expected tracker cohort ID is invalid")
    calibration_sha = _sha(
        expected_calibration_sha256, "expected tracker calibration SHA-256"
    )
    recipe_sha = _sha(
        expected_recipe_artifact_sha256, "expected tracker recipe SHA-256"
    )
    if not isinstance(image_bindings, Mapping):
        raise TrackerReceiptAdapterError("trusted tracker image bindings are missing")
    if (
        isinstance(receipts, (str, bytes))
        or not isinstance(receipts, Sequence)
        or len(receipts) != len(vials)
    ):
        raise TrackerReceiptAdapterError(
            "tracker receipt denominator must match recipe cases"
        )
    validator = _producer_validator()
    if not callable(validator):
        raise TrackerReceiptAdapterError("tracker producer validator is invalid")
    checked_receipts = [_validate_receipt(value, validator) for value in receipts]
    by_vial_id: dict[str, Mapping[str, object]] = {}
    for checked in checked_receipts:
        vial_id = checked.get("vial_id")
        if not isinstance(vial_id, str) or not vial_id or vial_id in by_vial_id:
            raise TrackerReceiptAdapterError("tracker receipt vial identity is invalid")
        by_vial_id[vial_id] = checked

    used_images: set[str] = set()
    plans: list[TrackerEndpointPlan] = []
    for vial in vials:
        case_id = str(vial["case_id"])
        expected_vial_id = vial_ids[case_id]
        if not isinstance(expected_vial_id, str) or not expected_vial_id:
            raise TrackerReceiptAdapterError("tracker vial identity map is invalid")
        receipt = by_vial_id.get(expected_vial_id)
        if receipt is None:
            raise TrackerReceiptAdapterError("tracker receipt vial order/identity mismatch")
        expected_receipt_sha = _sha(
            receipt_hashes[case_id], f"expected receipt SHA-256 for {case_id}"
        )
        source_sha = _sha(
            source_hashes[case_id], f"expected source SHA-256 for {case_id}"
        )
        if receipt.get("cohort_id") != expected_cohort_id:
            raise TrackerReceiptAdapterError("tracker cohort identity mismatch")
        if receipt.get("receipt_sha256") != expected_receipt_sha:
            raise TrackerReceiptAdapterError("tracker receipt SHA-256 binding mismatch")
        if receipt.get("source_artifact_sha256") != source_sha:
            raise TrackerReceiptAdapterError("tracker source SHA-256 binding mismatch")
        if receipt.get("calibration_sha256") != calibration_sha:
            raise TrackerReceiptAdapterError("tracker calibration SHA-256 binding mismatch")
        if receipt.get("recipe_artifact_sha256") != recipe_sha:
            raise TrackerReceiptAdapterError("tracker recipe SHA-256 binding mismatch")

        order = vial.get("dispense_order")
        programmed = vial.get("programmed_tank_ml")
        stages = receipt.get("stages")
        if (
            not isinstance(order, list)
            or not isinstance(programmed, Mapping)
            or not isinstance(stages, list)
            or len(stages) != len(order)
        ):
            raise TrackerReceiptAdapterError("tracker and recipe stage denominator differs")
        cumulative = 0.0
        projected: list[TrackerEndpointStage] = []
        for ordinal, (tank_id, stage) in enumerate(zip(order, stages, strict=True), 1):
            if not isinstance(tank_id, str) or not isinstance(stage, Mapping):
                raise TrackerReceiptAdapterError("tracker recipe stage is invalid")
            programmed_ml = _number(
                programmed.get(tank_id), f"recipe stage {tank_id} programmed mL"
            )
            cumulative += programmed_ml
            try:
                stage_dispense_ml = float(stage.get("expected_dispense_ml", -1))
                stage_cumulative_ml = float(stage.get("expected_cumulative_ml", -1))
            except (TypeError, ValueError) as error:
                raise TrackerReceiptAdapterError(
                    "tracker and recipe stage differ"
                ) from error
            if (
                stage.get("fill_step") != ordinal
                or stage.get("stream_step_index") != ordinal
                or stage.get("tank_id") != tank_id
                or stage_dispense_ml != programmed_ml
                or stage_cumulative_ml != cumulative
            ):
                raise TrackerReceiptAdapterError("tracker and recipe stage differ")
            if (
                stage.get("agreement") is not True
                or stage.get("tank_cluster_consistent") is not True
            ):
                raise TrackerReceiptAdapterError(
                    "tracker stage lacks accepted agreement"
                )
            volume = _number(
                stage.get("twin_geometry_equivalent_volume_ml"),
                "tracker twin-geometry-equivalent volume",
            )
            images = stage.get("images")
            if not isinstance(images, Mapping) or set(images) != {"early", "mid", "end"}:
                raise TrackerReceiptAdapterError("tracker stage image roles are invalid")
            image_ids = tuple(images[role] for role in ("early", "mid", "end"))
            if (
                any(not isinstance(image_id, str) or not image_id for image_id in image_ids)
                or len(set(image_ids)) != 3
                or used_images.intersection(image_ids)
            ):
                raise TrackerReceiptAdapterError(
                    "tracker early/mid/end images must be distinct"
                )
            frame_indices: list[int] = []
            for image_id in image_ids:
                binding = image_bindings.get(image_id)
                if not isinstance(binding, Mapping) or set(binding) != {
                    "case_id", "frame_index", "rgb_sha256", "mask_sha256"
                }:
                    raise TrackerReceiptAdapterError(
                        "tracker image lacks a trusted registry binding"
                    )
                if binding.get("case_id") != case_id:
                    raise TrackerReceiptAdapterError("tracker image case binding differs")
                frame_index = binding.get("frame_index")
                if type(frame_index) is not int or frame_index < 0:
                    raise TrackerReceiptAdapterError("tracker image frame binding is invalid")
                _sha(binding.get("rgb_sha256"), "tracker image RGB SHA-256")
                _sha(binding.get("mask_sha256"), "tracker image mask SHA-256")
                frame_indices.append(frame_index)
            if not frame_indices[0] < frame_indices[1] < frame_indices[2]:
                raise TrackerReceiptAdapterError(
                    "tracker early/mid/end frame order is invalid"
                )
            used_images.update(image_ids)
            projected.append(TrackerEndpointStage(
                fill_step=ordinal,
                tank_id=tank_id,
                expected_dispense_ml=programmed_ml,
                expected_cumulative_ml=cumulative,
                twin_geometry_equivalent_volume_ml=volume,
                agreement=True,
                stream_lab=_triplet(stage.get("stream_lab"), "tracker stream Lab"),
                stream_spread_lab=_triplet(
                    stage.get("stream_spread_lab"), "tracker stream spread Lab"
                ),
                early_image_id=str(image_ids[0]),
                mid_image_id=str(image_ids[1]),
                end_image_id=str(image_ids[2]),
                early_frame_index=frame_indices[0],
                mid_frame_index=frame_indices[1],
                end_frame_index=frame_indices[2],
            ))
        plans.append(TrackerEndpointPlan(
            case_id=case_id,
            vial_id=expected_vial_id,
            cohort_id=expected_cohort_id,
            receipt_sha256=expected_receipt_sha,
            source_artifact_sha256=source_sha,
            calibration_sha256=calibration_sha,
            recipe_artifact_sha256=recipe_sha,
            stages=tuple(projected),
        ))
    return tuple(plans)


__all__ = [
    "TrackerEndpointPlan",
    "TrackerEndpointStage",
    "TrackerReceiptAdapterError",
    "adapt_tracker_receipts_to_endpoint_plans",
]

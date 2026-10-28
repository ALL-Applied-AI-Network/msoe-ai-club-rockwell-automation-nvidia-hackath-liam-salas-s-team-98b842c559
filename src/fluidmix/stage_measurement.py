"""Evidence-only stage colour measurements against explicit in-recipe peers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StageImage:
    """One vial image after a known dispense stage."""

    record_id: str
    image_id: str
    run_id: str
    recipe_id: str
    data_origin: str
    split_role: str
    vial_id: str
    stage_id: str
    stage_ordinal: int
    stage_tank_id: str
    dispense_order: tuple[str, ...]
    rgb: Any
    liquid_mask: Any


def measure_stage_evidence(
    target: StageImage,
    references: list[StageImage] | tuple[StageImage, ...],
    *,
    minimum_reference_count: int,
) -> dict[str, object]:
    """Return quantitative evidence; decisions remain a downstream responsibility."""

    if (
        isinstance(minimum_reference_count, bool)
        or not isinstance(minimum_reference_count, int)
        or minimum_reference_count <= 0
    ):
        raise ValueError("minimum_reference_count must be a positive integer")
    if not isinstance(target, StageImage):
        raise TypeError("target must be a StageImage")
    if not isinstance(references, (list, tuple)):
        raise TypeError("references must be a list or tuple")
    if any(not isinstance(reference, StageImage) for reference in references):
        raise TypeError("every reference must be a StageImage")

    try:
        measured_target = _measure(target)
    except _EvidenceError as exc:
        return _insufficient(
            target,
            minimum_reference_count,
            [f"target:{exc.code}"],
        )

    measured_references: list[dict[str, object]] = []
    codes: list[str] = []
    seen_record_ids: set[str] = set()
    seen_image_ids: set[str] = set()
    seen_vial_ids: set[str] = set()
    seen_rgb_hashes: set[str] = set()
    for reference in references:
        context_code = _context_mismatch(target, reference)
        if context_code is not None:
            codes.append(f"reference:{reference.record_id}:{context_code}")
            continue
        if reference.record_id == target.record_id:
            codes.append("target_record_reused_as_reference")
            continue
        if reference.image_id == target.image_id:
            codes.append("target_image_reused_as_reference")
            continue
        if reference.vial_id == target.vial_id:
            codes.append("target_vial_reused_as_reference")
            continue
        if (
            reference.record_id in seen_record_ids
            or reference.image_id in seen_image_ids
            or reference.vial_id in seen_vial_ids
        ):
            codes.append("duplicate_reference_identity")
            continue
        try:
            measured = _measure(reference)
        except _EvidenceError as exc:
            codes.append(f"reference:{reference.record_id}:{exc.code}")
            continue
        if measured["rgb_pixel_sha256"] == measured_target["rgb_pixel_sha256"]:
            codes.append("target_rgb_bytes_reused_as_reference")
            continue
        if measured["rgb_pixel_sha256"] in seen_rgb_hashes:
            codes.append("duplicate_reference_rgb_bytes")
            continue
        seen_record_ids.add(reference.record_id)
        seen_image_ids.add(reference.image_id)
        seen_vial_ids.add(reference.vial_id)
        seen_rgb_hashes.add(str(measured["rgb_pixel_sha256"]))
        measured_references.append(measured)

    measured_references.sort(key=lambda row: str(row["record_id"]))

    if codes:
        return _insufficient(
            target,
            minimum_reference_count,
            list(dict.fromkeys(codes)),
            measured_target=measured_target,
            measured_references=measured_references,
        )
    if len(measured_references) < minimum_reference_count:
        return _insufficient(
            target,
            minimum_reference_count,
            ["reference_count_below_minimum"],
            measured_target=measured_target,
            measured_references=measured_references,
        )

    labs = np.asarray(
        [reference["lab"] for reference in measured_references],
        dtype=np.float64,
    )
    center = np.median(labs, axis=0)
    radii = np.linalg.norm(labs - center, axis=1)
    pairwise = [
        float(np.linalg.norm(labs[left] - labs[right]))
        for left in range(len(labs))
        for right in range(left + 1, len(labs))
    ]
    if not all(
        math.isfinite(value)
        for value in [*center.tolist(), *radii.tolist(), *pairwise]
    ):
        return _insufficient(
            target,
            minimum_reference_count,
            ["reference_spread_nonfinite"],
            measured_target=measured_target,
            measured_references=measured_references,
        )
    median_radius = float(np.median(radii))
    max_radius = float(np.max(radii))
    mad_radius = float(np.median(np.abs(radii - median_radius)))
    pairwise_diameter = max(pairwise, default=0.0)
    target_delta = float(
        np.linalg.norm(np.asarray(measured_target["lab"], dtype=np.float64) - center)
    )
    numeric = [
        median_radius,
        max_radius,
        mad_radius,
        pairwise_diameter,
        target_delta,
    ]
    if not all(math.isfinite(value) for value in numeric):
        return _insufficient(
            target,
            minimum_reference_count,
            ["reference_spread_nonfinite"],
            measured_target=measured_target,
            measured_references=measured_references,
        )

    return {
        "validity": "valid",
        "limit_codes": [],
        "unit": "delta_e76",
        "target": measured_target,
        "reference": _reference_summary(
            target,
            measured_references,
            minimum_reference_count,
            center=center.tolist(),
            median_radius=median_radius,
            max_radius=max_radius,
            mad_radius=mad_radius,
            pairwise_diameter=pairwise_diameter,
        ),
        "target_delta_e76": target_delta,
    }


class _EvidenceError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _EvidenceError(f"{field}_invalid")
    return value


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _measure(record: StageImage) -> dict[str, object]:
    for field in (
        "record_id",
        "image_id",
        "run_id",
        "recipe_id",
        "data_origin",
        "split_role",
        "vial_id",
        "stage_id",
        "stage_tank_id",
    ):
        _nonempty_text(getattr(record, field), field)
    if (
        isinstance(record.stage_ordinal, bool)
        or not isinstance(record.stage_ordinal, int)
        or record.stage_ordinal <= 0
    ):
        raise _EvidenceError("stage_ordinal_invalid")
    if (
        not isinstance(record.dispense_order, tuple)
        or not record.dispense_order
        or any(not isinstance(tank, str) or not tank for tank in record.dispense_order)
        or len(set(record.dispense_order)) != len(record.dispense_order)
    ):
        raise _EvidenceError("dispense_order_invalid")
    if record.stage_ordinal > len(record.dispense_order):
        raise _EvidenceError("stage_ordinal_out_of_range")
    if record.dispense_order[record.stage_ordinal - 1] != record.stage_tank_id:
        raise _EvidenceError("stage_tank_order_mismatch")
    if not isinstance(record.rgb, np.ndarray):
        raise _EvidenceError("rgb_not_array")
    if record.rgb.dtype != np.uint8 or record.rgb.ndim != 3 or record.rgb.shape[2] != 3:
        raise _EvidenceError("rgb_shape_or_dtype_invalid")
    if not isinstance(record.liquid_mask, np.ndarray):
        raise _EvidenceError("mask_not_array")
    if record.liquid_mask.dtype != np.uint8 or record.liquid_mask.shape != record.rgb.shape[:2]:
        raise _EvidenceError("mask_shape_or_dtype_invalid")
    rgb = np.array(record.rgb, dtype=np.uint8, copy=True)
    mask = np.array(record.liquid_mask, dtype=np.uint8, copy=True)
    mask_values = set(int(value) for value in np.unique(mask))
    if not mask_values.issubset({0, 255}):
        raise _EvidenceError("mask_not_binary")
    selected = mask == 255
    liquid_pixels = int(np.count_nonzero(selected))
    if liquid_pixels <= 0:
        raise _EvidenceError("mask_empty")

    from skimage.color import rgb2lab

    lab_image = rgb2lab(rgb.astype(np.float64) / 255.0)
    lab = [
        float(np.median(lab_image[..., channel][selected]))
        for channel in range(3)
    ]
    if not all(math.isfinite(value) for value in lab):
        raise _EvidenceError("lab_nonfinite")
    return {
        "record_id": record.record_id,
        "image_id": record.image_id,
        "run_id": record.run_id,
        "recipe_id": record.recipe_id,
        "data_origin": record.data_origin,
        "split_role": record.split_role,
        "vial_id": record.vial_id,
        "stage_id": record.stage_id,
        "stage_ordinal": record.stage_ordinal,
        "stage_tank_id": record.stage_tank_id,
        "dispense_order": list(record.dispense_order),
        "rgb_pixel_sha256": _array_sha256(rgb),
        "mask_pixel_sha256": _array_sha256(mask),
        "liquid_pixel_count": liquid_pixels,
        "lab": lab,
        "lab_unit": "cie_lab_d65_2deg",
    }


def _context_mismatch(target: StageImage, reference: StageImage) -> str | None:
    for field, code in (
        ("data_origin", "data_origin_mismatch"),
        ("recipe_id", "recipe_mismatch"),
        ("split_role", "split_mismatch"),
        ("stage_id", "stage_id_mismatch"),
        ("stage_ordinal", "stage_ordinal_mismatch"),
        ("stage_tank_id", "stage_tank_mismatch"),
        ("dispense_order", "dispense_order_mismatch"),
    ):
        if getattr(reference, field) != getattr(target, field):
            return code
    return None


def _reference_summary(
    target: StageImage,
    measured_references: list[dict[str, object]],
    minimum_reference_count: int,
    *,
    center: list[float] | None = None,
    median_radius: float | None = None,
    max_radius: float | None = None,
    mad_radius: float | None = None,
    pairwise_diameter: float | None = None,
) -> dict[str, object]:
    return {
        "minimum_count": minimum_reference_count,
        "count": len(measured_references),
        "record_ids": [str(row["record_id"]) for row in measured_references],
        "image_ids": [str(row["image_id"]) for row in measured_references],
        "run_ids": [str(row["run_id"]) for row in measured_references],
        "same_run_count": sum(row["run_id"] == target.run_id for row in measured_references),
        "other_run_count": sum(row["run_id"] != target.run_id for row in measured_references),
        "center_lab": center,
        "center_lab_unit": "cie_lab_d65_2deg",
        "median_radius_delta_e76": median_radius,
        "max_radius_delta_e76": max_radius,
        "mad_radius_delta_e76": mad_radius,
        "pairwise_diameter_delta_e76": pairwise_diameter,
        "unit": "delta_e76",
    }


def _insufficient(
    target: StageImage,
    minimum_reference_count: int,
    codes: list[str],
    *,
    measured_target: dict[str, object] | None = None,
    measured_references: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "validity": "insufficient_evidence",
        "limit_codes": list(dict.fromkeys(codes)),
        "unit": "delta_e76",
        "target": measured_target,
        "reference": _reference_summary(
            target,
            measured_references or [],
            minimum_reference_count,
        ),
        "target_delta_e76": None,
    }

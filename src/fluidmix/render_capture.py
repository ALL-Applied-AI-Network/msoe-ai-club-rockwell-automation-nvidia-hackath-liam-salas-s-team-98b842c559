"""Fail-closed semantic planning and shared-capture validation for FillCam renders.

The contracts in this module are deliberately independent of Isaac Sim.  They validate
reviewed USD facts and already-captured arrays; they do not prove that a session-layer edit
was applied, that geometry rendered correctly, or that synthetic images transfer to a real
camera.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
import re
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SemanticOverrideSpec:
    """The one reviewed liquid subtree and its required class."""

    root_path: str = "/FillStation/FillStation_LocalRoot/Fill_liquid"
    mesh_path: str = "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid"
    class_name: str = "liquid"


LIQUID_OVERRIDE = SemanticOverrideSpec()


@dataclass(frozen=True)
class TransparentAnnotationOverrideSpec:
    """The one reviewed transparent-vial segmentation probe."""

    prim_path: str = "/FillStation/FillStation_LocalRoot/Fill_vial_active/Mesh_998"
    parent_path: str = "/FillStation/FillStation_LocalRoot/Fill_vial_active"
    required_parent_class: str = "vial"
    attribute_name: str = "primvars:doNotCastShadows"
    authored_value: bool = True


TRANSPARENT_ANNOTATION_OVERRIDE = TransparentAnnotationOverrideSpec()


@dataclass(frozen=True)
class TransparentAnnotationSourceFact:
    attribute_exists: bool
    type_name: str | None
    has_authored_value: bool
    value: bool | None


@dataclass(frozen=True)
class TransparentAnnotationPlan:
    override: TransparentAnnotationOverrideSpec
    prim_type: str
    planned_parent_class: str
    source: TransparentAnnotationSourceFact


class TransparentAnnotationPlanError(ValueError):
    """Raised when the fixed transparent-vial probe is not supported by facts."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


def plan_transparent_annotation_override(
    prim_types: Mapping[str, object],
    planned_classes: Mapping[str, object],
    source: TransparentAnnotationSourceFact,
) -> TransparentAnnotationPlan:
    """Validate facts for the one fixed transparent-vial annotation probe."""

    override = TRANSPARENT_ANNOTATION_OVERRIDE
    if override.prim_path not in prim_types:
        raise TransparentAnnotationPlanError(
            "transparent_vial_mesh_missing", override.prim_path
        )
    prim_type = prim_types[override.prim_path]
    if prim_type != "Mesh":
        raise TransparentAnnotationPlanError(
            "transparent_vial_mesh_wrong_type",
            f"{override.prim_path} is {prim_type!r}, expected 'Mesh'",
        )
    if override.parent_path not in prim_types:
        raise TransparentAnnotationPlanError(
            "transparent_vial_parent_missing", override.parent_path
        )
    parent_class = planned_classes.get(override.parent_path)
    if parent_class != override.required_parent_class:
        raise TransparentAnnotationPlanError(
            "transparent_vial_parent_class_invalid",
            f"{override.parent_path} planned class is {parent_class!r}, expected "
            f"{override.required_parent_class!r}",
        )
    if override.prim_path in planned_classes:
        raise TransparentAnnotationPlanError(
            "transparent_vial_mesh_class_conflict",
            f"{override.prim_path} has a direct planned class "
            f"{planned_classes[override.prim_path]!r}",
        )
    if not isinstance(source, TransparentAnnotationSourceFact):
        raise TransparentAnnotationPlanError(
            "transparent_annotation_source_fact_conflict",
            f"source fact has wrong type: {source!r}",
        )
    if (
        type(source.attribute_exists) is not bool
        or type(source.has_authored_value) is not bool
    ):
        raise TransparentAnnotationPlanError(
            "transparent_annotation_source_fact_conflict",
            f"source presence facts are not bool: {source!r}",
        )
    if source.attribute_exists:
        if source.type_name != "bool":
            raise TransparentAnnotationPlanError(
                "transparent_annotation_attribute_conflict",
                f"{override.attribute_name} has type {source.type_name!r}, expected 'bool'",
            )
        if source.value is not None and type(source.value) is not bool:
            raise TransparentAnnotationPlanError(
                "transparent_annotation_source_fact_conflict",
                f"{override.attribute_name} has non-Bool value {source.value!r}",
            )
    elif (
        source.type_name is not None
        or source.has_authored_value
        or source.value is not None
    ):
        raise TransparentAnnotationPlanError(
            "transparent_annotation_source_fact_conflict",
            f"absent {override.attribute_name} has contradictory source facts",
        )
    return TransparentAnnotationPlan(
        override=override,
        prim_type=prim_type,
        planned_parent_class=parent_class,
        source=source,
    )


def transparent_annotation_summary_record(
    plan: TransparentAnnotationPlan,
    *,
    composed_type_name: object,
    composed_value: object,
) -> dict[str, object]:
    """Serialize verified source/composed facts for summary schema 1."""

    if not isinstance(plan, TransparentAnnotationPlan):
        raise TransparentAnnotationPlanError(
            "transparent_annotation_plan_invalid", repr(plan)
        )
    if plan.override != TRANSPARENT_ANNOTATION_OVERRIDE:
        raise TransparentAnnotationPlanError(
            "transparent_annotation_plan_invalid", "override is not the fixed reviewed spec"
        )
    plan_transparent_annotation_override(
        {
            plan.override.parent_path: "Xform",
            plan.override.prim_path: plan.prim_type,
        },
        {plan.override.parent_path: plan.planned_parent_class},
        plan.source,
    )
    if composed_type_name != "bool" or type(composed_value) is not bool:
        raise TransparentAnnotationPlanError(
            "transparent_annotation_composed_verification_failed",
            f"composed type/value are {composed_type_name!r}/{composed_value!r}",
        )
    if composed_value is not plan.override.authored_value:
        raise TransparentAnnotationPlanError(
            "transparent_annotation_composed_verification_failed",
            f"composed value is {composed_value!r}, expected "
            f"{plan.override.authored_value!r}",
        )
    return {
        "prim_path": plan.override.prim_path,
        "parent_path": plan.override.parent_path,
        "required_prim_type": "Mesh",
        "required_parent_class": plan.override.required_parent_class,
        "attribute_name": plan.override.attribute_name,
        "authored_value": plan.override.authored_value,
        "session_layer_only": True,
        "source": {
            "prim_type": plan.prim_type,
            "planned_parent_class": plan.planned_parent_class,
            "attribute_exists": plan.source.attribute_exists,
            "type_name": plan.source.type_name,
            "has_authored_value": plan.source.has_authored_value,
            "value": plan.source.value,
        },
        "composed": {
            "type_name": composed_type_name,
            "value": composed_value,
        },
    }


@dataclass(frozen=True)
class SemanticAssignment:
    path: str
    class_name: str


@dataclass(frozen=True)
class SemanticPlan:
    copied_assignments: tuple[SemanticAssignment, ...]
    liquid_assignment: SemanticAssignment
    excluded_subtree: tuple[str, ...]

    @property
    def assignments(self) -> tuple[SemanticAssignment, ...]:
        return self.copied_assignments + (self.liquid_assignment,)


class SemanticPlanError(ValueError):
    """Raised when reviewed prim facts cannot support an unambiguous override."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


@dataclass(frozen=True)
class ClassMappingRecord:
    semantic_id: int
    labels: tuple[str, ...]


@dataclass(frozen=True)
class ClassIdResolution:
    records: tuple[ClassMappingRecord, ...]

    def ids_for(self, class_name: str) -> tuple[int, ...]:
        return tuple(
            record.semantic_id for record in self.records if class_name in record.labels
        )

    @property
    def liquid_ids(self) -> tuple[int, ...]:
        return self.ids_for(LIQUID_OVERRIDE.class_name)

    @property
    def vial_ids(self) -> tuple[int, ...]:
        return self.ids_for("vial")


class LabelMappingError(ValueError):
    """Raised when an annotator label map is ambiguous or malformed."""

    def __init__(self, reason_code: str, detail: str):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code


@dataclass(frozen=True)
class InstanceMappingRecord:
    instance_id: int
    path: str


@dataclass(frozen=True)
class ArrayStats:
    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    std: float


@dataclass(frozen=True)
class ClassPixelCount:
    class_name: str
    class_ids: tuple[int, ...]
    pixel_count: int


@dataclass(frozen=True)
class CaptureValidation:
    valid: bool
    reason_codes: tuple[str, ...]
    raster_shape: tuple[int, int]
    rgb: ArrayStats | None
    semantic: ArrayStats | None
    instance: ArrayStats | None
    semantic_ids: tuple[int, ...]
    instance_ids: tuple[int, ...]
    class_mapping: ClassIdResolution | None
    instance_mapping: tuple[InstanceMappingRecord, ...]
    class_pixel_counts: tuple[ClassPixelCount, ...]
    selected_liquid_instance_ids: tuple[int, ...]
    liquid_instance_pixel_count: int
    instance_agreement: bool

    def pixel_count_for(self, class_name: str) -> int:
        for record in self.class_pixel_counts:
            if record.class_name == class_name:
                return record.pixel_count
        raise KeyError(class_name)


def resolve_class_ids(id_to_labels: Mapping[object, object]) -> ClassIdResolution:
    """Resolve semantic class IDs from a non-colorized annotator mapping."""

    if not isinstance(id_to_labels, Mapping):
        raise LabelMappingError("semantic_mapping_not_mapping", repr(id_to_labels))

    records: list[ClassMappingRecord] = []
    seen_ids: set[int] = set()
    for raw_id, raw_record in id_to_labels.items():
        semantic_id = _normalize_annotator_id(raw_id, kind="semantic")
        if semantic_id in seen_ids:
            raise LabelMappingError(
                "semantic_id_collision", f"multiple keys normalize to {semantic_id}"
            )
        seen_ids.add(semantic_id)

        if not isinstance(raw_record, Mapping) or "class" not in raw_record:
            raise LabelMappingError(
                "semantic_label_record_malformed",
                f"ID {semantic_id} has no class mapping: {raw_record!r}",
            )
        raw_labels = raw_record["class"]
        if not isinstance(raw_labels, str):
            raise LabelMappingError(
                "semantic_label_record_malformed",
                f"ID {semantic_id} class is not a string: {raw_labels!r}",
            )
        labels = tuple(part.strip() for part in raw_labels.split(","))
        if not labels or any(not part for part in labels):
            raise LabelMappingError(
                "semantic_label_record_malformed",
                f"ID {semantic_id} has empty class component: {raw_labels!r}",
            )
        records.append(ClassMappingRecord(semantic_id=semantic_id, labels=labels))

    return ClassIdResolution(tuple(sorted(records, key=lambda record: record.semantic_id)))


def _normalize_annotator_id(raw_id: object, *, kind: str) -> int:
    if isinstance(raw_id, bool):
        raise LabelMappingError(f"{kind}_id_malformed", repr(raw_id))
    if isinstance(raw_id, Integral):
        normalized = int(raw_id)
    elif isinstance(raw_id, str) and re.fullmatch(r"[0-9]+", raw_id):
        normalized = int(raw_id, 10)
    else:
        raise LabelMappingError(f"{kind}_id_malformed", repr(raw_id))
    if normalized < 0:
        raise LabelMappingError(f"{kind}_id_malformed", repr(raw_id))
    return normalized


def _resolve_instance_paths(id_to_labels: object) -> tuple[InstanceMappingRecord, ...]:
    if not isinstance(id_to_labels, Mapping):
        raise LabelMappingError("instance_mapping_not_mapping", repr(id_to_labels))
    records: list[InstanceMappingRecord] = []
    seen_ids: set[int] = set()
    for raw_id, raw_path in id_to_labels.items():
        instance_id = _normalize_annotator_id(raw_id, kind="instance")
        if instance_id in seen_ids:
            raise LabelMappingError(
                "instance_id_collision", f"multiple keys normalize to {instance_id}"
            )
        seen_ids.add(instance_id)
        if not isinstance(raw_path, str) or not raw_path:
            raise LabelMappingError(
                "instance_label_record_malformed",
                f"ID {instance_id} path is invalid: {raw_path!r}",
            )
        records.append(InstanceMappingRecord(instance_id=instance_id, path=raw_path))
    return tuple(sorted(records, key=lambda record: record.instance_id))


def _array_stats(array: np.ndarray) -> ArrayStats:
    return ArrayStats(
        shape=tuple(int(size) for size in array.shape),
        dtype=str(array.dtype),
        minimum=float(array.min()),
        maximum=float(array.max()),
        mean=float(array.mean()),
        std=float(array.std()),
    )


def validate_capture(
    rgb: object,
    semantic: object,
    instance: object,
    semantic_id_to_labels: object,
    instance_id_to_labels: object,
    *,
    raster_shape: tuple[int, int],
    require_liquid_pixels: bool,
    override: SemanticOverrideSpec = LIQUID_OVERRIDE,
) -> CaptureValidation:
    """Validate RGB, semantic IDs, and instance IDs from one render step."""

    reasons: list[str] = []
    if (
        not isinstance(raster_shape, tuple)
        or len(raster_shape) != 2
        or any(isinstance(size, bool) or not isinstance(size, Integral) or size <= 0 for size in raster_shape)
    ):
        raster_shape = (0, 0)
        reasons.append("raster_shape_invalid")
    else:
        raster_shape = (int(raster_shape[0]), int(raster_shape[1]))
    if not isinstance(require_liquid_pixels, bool):
        reasons.append("require_liquid_pixels_invalid")

    arrays = (("rgb", rgb), ("semantic", semantic), ("instance", instance))
    for name, array in arrays:
        if not isinstance(array, np.ndarray):
            reasons.append(f"{name}_not_array")

    expected_rgb_shape = raster_shape + (3,)
    if isinstance(rgb, np.ndarray) and rgb.shape != expected_rgb_shape:
        reasons.append("rgb_shape_invalid")
    if isinstance(semantic, np.ndarray) and semantic.shape != raster_shape:
        reasons.append("semantic_shape_invalid")
    if isinstance(instance, np.ndarray) and instance.shape != raster_shape:
        reasons.append("instance_shape_invalid")

    if isinstance(rgb, np.ndarray) and rgb.dtype != np.dtype(np.uint8):
        reasons.append("rgb_dtype_invalid")
    for name, array in (("semantic", semantic), ("instance", instance)):
        if isinstance(array, np.ndarray) and (
            np.issubdtype(array.dtype, np.bool_)
            or not np.issubdtype(array.dtype, np.integer)
        ):
            reasons.append(f"{name}_dtype_invalid")

    structural_reasons = {
        "raster_shape_invalid",
        "rgb_not_array",
        "semantic_not_array",
        "instance_not_array",
        "rgb_shape_invalid",
        "semantic_shape_invalid",
        "instance_shape_invalid",
        "rgb_dtype_invalid",
        "semantic_dtype_invalid",
        "instance_dtype_invalid",
    }
    if structural_reasons.intersection(reasons):
        return CaptureValidation(
            valid=False,
            reason_codes=tuple(dict.fromkeys(reasons)),
            raster_shape=raster_shape,
            rgb=None,
            semantic=None,
            instance=None,
            semantic_ids=(),
            instance_ids=(),
            class_mapping=None,
            instance_mapping=(),
            class_pixel_counts=(),
            selected_liquid_instance_ids=(),
            liquid_instance_pixel_count=0,
            instance_agreement=False,
        )

    assert isinstance(rgb, np.ndarray)
    assert isinstance(semantic, np.ndarray)
    assert isinstance(instance, np.ndarray)
    if not np.any(rgb != rgb[0, 0]):
        reasons.append("rgb_spatially_constant")

    try:
        class_mapping = resolve_class_ids(semantic_id_to_labels)  # type: ignore[arg-type]
    except (LabelMappingError, TypeError):
        class_mapping = None
        reasons.append("semantic_mapping_invalid")
    try:
        instance_mapping = _resolve_instance_paths(instance_id_to_labels)
    except (LabelMappingError, TypeError):
        instance_mapping = ()
        reasons.append("instance_mapping_invalid")

    semantic_ids = tuple(int(value) for value in np.unique(semantic))
    instance_ids = tuple(int(value) for value in np.unique(instance))
    if any(value < 0 for value in semantic_ids):
        reasons.append("semantic_id_negative")
    if any(value < 0 for value in instance_ids):
        reasons.append("instance_id_negative")

    mapped_semantic_ids = (
        {record.semantic_id for record in class_mapping.records}
        if class_mapping is not None
        else set()
    )
    mapped_instance_ids = {record.instance_id for record in instance_mapping}
    if class_mapping is not None and not set(semantic_ids).issubset(mapped_semantic_ids):
        reasons.append("semantic_id_unmapped")
    if not set(instance_ids).issubset(mapped_instance_ids):
        reasons.append("instance_id_unmapped")

    liquid_ids = class_mapping.liquid_ids if class_mapping is not None else ()
    vial_ids = class_mapping.vial_ids if class_mapping is not None else ()
    if set(liquid_ids).intersection(vial_ids):
        reasons.append("class_ids_overlap")

    liquid_mask = np.isin(semantic, liquid_ids)
    vial_mask = np.isin(semantic, vial_ids)
    liquid_count = int(np.count_nonzero(liquid_mask))
    vial_count = int(np.count_nonzero(vial_mask))
    total_pixels = int(raster_shape[0] * raster_shape[1])
    if require_liquid_pixels:
        if liquid_count == 0:
            reasons.append("liquid_pixels_absent")
        elif liquid_count == total_pixels:
            reasons.append("liquid_pixels_whole_frame")
        if vial_count == 0:
            reasons.append("vial_pixels_absent")
        elif vial_count == total_pixels:
            reasons.append("vial_pixels_whole_frame")

    allowed_paths = {override.root_path, override.mesh_path}
    instance_path_by_id = {
        record.instance_id: record.path for record in instance_mapping
    }
    selected_liquid_instance_ids = tuple(
        sorted(
            instance_id
            for instance_id in instance_ids
            if instance_path_by_id.get(instance_id) in allowed_paths
        )
    )
    if liquid_count:
        liquid_region_instance_ids = {
            int(value) for value in np.unique(instance[liquid_mask])
        }
        if any(
            instance_path_by_id.get(instance_id) not in allowed_paths
            for instance_id in liquid_region_instance_ids
            if instance_id in instance_path_by_id
        ):
            reasons.append("liquid_instance_path_invalid")
    instance_liquid_mask = np.isin(instance, selected_liquid_instance_ids)
    liquid_instance_pixel_count = int(np.count_nonzero(instance_liquid_mask))
    instance_agreement = bool(np.array_equal(instance_liquid_mask, liquid_mask))
    if not instance_agreement:
        reasons.append("liquid_instance_mismatch")

    reason_codes = tuple(dict.fromkeys(reasons))
    return CaptureValidation(
        valid=not reason_codes,
        reason_codes=reason_codes,
        raster_shape=raster_shape,
        rgb=_array_stats(rgb),
        semantic=_array_stats(semantic),
        instance=_array_stats(instance),
        semantic_ids=semantic_ids,
        instance_ids=instance_ids,
        class_mapping=class_mapping,
        instance_mapping=instance_mapping,
        class_pixel_counts=(
            ClassPixelCount("liquid", liquid_ids, liquid_count),
            ClassPixelCount("vial", vial_ids, vial_count),
        ),
        selected_liquid_instance_ids=selected_liquid_instance_ids,
        liquid_instance_pixel_count=liquid_instance_pixel_count,
        instance_agreement=instance_agreement,
    )


def plan_semantic_assignments(
    source_classes: Mapping[str, object],
    prim_types: Mapping[str, object],
    existing_effective_classes: Mapping[str, object],
    *,
    override: SemanticOverrideSpec = LIQUID_OVERRIDE,
) -> SemanticPlan:
    """Plan source-label copies plus the reviewed liquid-root override."""

    if override.root_path not in prim_types:
        raise SemanticPlanError("liquid_root_missing", override.root_path)
    if override.mesh_path not in prim_types:
        raise SemanticPlanError("liquid_mesh_missing", override.mesh_path)
    if prim_types[override.mesh_path] != "Mesh":
        raise SemanticPlanError(
            "liquid_mesh_wrong_type",
            f"{override.mesh_path} is {prim_types[override.mesh_path]!r}, expected 'Mesh'",
        )

    def is_liquid_subtree(path: str) -> bool:
        return path == override.root_path or path.startswith(override.root_path + "/")

    excluded = tuple(sorted(path for path in prim_types if is_liquid_subtree(path)))
    for path, labels in existing_effective_classes.items():
        if not is_liquid_subtree(path) or labels is None:
            continue
        if isinstance(labels, str):
            present = bool(labels)
        elif isinstance(labels, Sequence):
            if not all(isinstance(label, str) and label for label in labels):
                raise SemanticPlanError(
                    "existing_class_fact_malformed",
                    f"{path} has invalid effective classes {labels!r}",
                )
            present = bool(labels)
        else:
            raise SemanticPlanError(
                "existing_class_fact_malformed",
                f"{path} has invalid effective classes {labels!r}",
            )
        if present:
            raise SemanticPlanError(
                "liquid_class_conflict",
                f"{path} already has effective class labels {labels!r}",
            )

    copied: list[SemanticAssignment] = []
    for path, value in source_classes.items():
        if is_liquid_subtree(path):
            continue
        if value is None:
            continue
        if not isinstance(value, str):
            raise SemanticPlanError(
                "source_class_malformed", f"{path} has non-string class {value!r}"
            )
        class_name = value.strip()
        if not class_name or class_name.casefold() == "none":
            continue
        copied.append(SemanticAssignment(path=path, class_name=class_name))

    return SemanticPlan(
        copied_assignments=tuple(sorted(copied, key=lambda item: item.path)),
        liquid_assignment=SemanticAssignment(override.root_path, override.class_name),
        excluded_subtree=excluded,
    )

"""CPU contracts for fail-closed FillCam semantic capture."""

import io
import json
import os
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import fluidmix.render_capture as render_capture

from fluidmix.render_capture import (
    LIQUID_OVERRIDE,
    TRANSPARENT_ANNOTATION_OVERRIDE,
    LabelMappingError,
    SemanticPlanError,
    TransparentAnnotationPlanError,
    TransparentAnnotationSourceFact,
    exact_instance_path_mask,
    plan_semantic_assignments,
    plan_transparent_annotation_override,
    resolve_class_ids,
    resolve_instance_paths,
    transparent_annotation_summary_record,
    validate_capture,
)


ROOT = LIQUID_OVERRIDE.root_path
MESH = LIQUID_OVERRIDE.mesh_path
RENDER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "twin_render_sequence.py"
VIAL_PARENT = TRANSPARENT_ANNOTATION_OVERRIDE.parent_path
VIAL_MESH = TRANSPARENT_ANNOTATION_OVERRIDE.prim_path


def _load_renderer_module():
    module_name = "fluidmix_test_twin_render_sequence"
    spec = importlib.util.spec_from_file_location(module_name, RENDER_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def test_transparent_annotation_plan_accepts_only_reviewed_vial_mesh_facts():
    source = TransparentAnnotationSourceFact(
        attribute_exists=False,
        type_name=None,
        has_authored_value=False,
        value=None,
    )

    plan = plan_transparent_annotation_override(
        {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
        {VIAL_PARENT: "vial"},
        source,
    )

    assert plan.override == TRANSPARENT_ANNOTATION_OVERRIDE
    assert plan.prim_type == "Mesh"
    assert plan.planned_parent_class == "vial"
    assert plan.source == source


@pytest.mark.parametrize(
    ("prim_types", "planned_classes", "source", "expected_reason"),
    [
        (
            {VIAL_PARENT: "Xform"},
            {VIAL_PARENT: "vial"},
            TransparentAnnotationSourceFact(False, None, False, None),
            "transparent_vial_mesh_missing",
        ),
        (
            {VIAL_PARENT: "Xform", VIAL_MESH: "Xform"},
            {VIAL_PARENT: "vial"},
            TransparentAnnotationSourceFact(False, None, False, None),
            "transparent_vial_mesh_wrong_type",
        ),
        (
            {VIAL_MESH: "Mesh"},
            {VIAL_PARENT: "vial"},
            TransparentAnnotationSourceFact(False, None, False, None),
            "transparent_vial_parent_missing",
        ),
        (
            {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
            {VIAL_PARENT: "carrier"},
            TransparentAnnotationSourceFact(False, None, False, None),
            "transparent_vial_parent_class_invalid",
        ),
        (
            {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
            {VIAL_PARENT: "vial", VIAL_MESH: "liquid"},
            TransparentAnnotationSourceFact(False, None, False, None),
            "transparent_vial_mesh_class_conflict",
        ),
        (
            {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
            {VIAL_PARENT: "vial"},
            TransparentAnnotationSourceFact(True, "token", True, True),
            "transparent_annotation_attribute_conflict",
        ),
        (
            {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
            {VIAL_PARENT: "vial"},
            TransparentAnnotationSourceFact(False, None, True, True),
            "transparent_annotation_source_fact_conflict",
        ),
        (
            {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
            {VIAL_PARENT: "vial"},
            object(),
            "transparent_annotation_source_fact_conflict",
        ),
    ],
)
def test_transparent_annotation_plan_rejects_missing_wrong_or_conflicting_facts(
    prim_types,
    planned_classes,
    source,
    expected_reason,
):
    with pytest.raises(TransparentAnnotationPlanError) as caught:
        plan_transparent_annotation_override(prim_types, planned_classes, source)

    assert caught.value.reason_code == expected_reason


def test_transparent_annotation_summary_records_source_and_composed_facts_truthfully():
    source = TransparentAnnotationSourceFact(
        attribute_exists=True,
        type_name="bool",
        has_authored_value=True,
        value=False,
    )
    plan = plan_transparent_annotation_override(
        {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
        {VIAL_PARENT: "vial"},
        source,
    )

    record = transparent_annotation_summary_record(
        plan,
        composed_type_name="bool",
        composed_value=True,
    )

    assert record == {
        "prim_path": VIAL_MESH,
        "parent_path": VIAL_PARENT,
        "required_prim_type": "Mesh",
        "required_parent_class": "vial",
        "attribute_name": "primvars:doNotCastShadows",
        "authored_value": True,
        "session_layer_only": True,
        "source": {
            "prim_type": "Mesh",
            "planned_parent_class": "vial",
            "attribute_exists": True,
            "type_name": "bool",
            "has_authored_value": True,
            "value": False,
        },
        "composed": {"type_name": "bool", "value": True},
    }


@pytest.mark.parametrize(
    ("composed_type_name", "composed_value"),
    [("token", True), ("bool", False), ("bool", 1)],
)
def test_transparent_annotation_summary_rejects_unverified_composed_fact(
    composed_type_name, composed_value
):
    plan = plan_transparent_annotation_override(
        {VIAL_PARENT: "Xform", VIAL_MESH: "Mesh"},
        {VIAL_PARENT: "vial"},
        TransparentAnnotationSourceFact(False, None, False, None),
    )

    with pytest.raises(TransparentAnnotationPlanError) as caught:
        transparent_annotation_summary_record(
            plan,
            composed_type_name=composed_type_name,
            composed_value=composed_value,
        )

    assert (
        caught.value.reason_code
        == "transparent_annotation_composed_verification_failed"
    )


def test_plan_preserves_vial_and_replaces_only_reviewed_liquid_subtree():
    vial = "/FillStation/Vial"
    nullish = {
        "/FillStation/Null": None,
        "/FillStation/Empty": "",
        "/FillStation/Whitespace": "  ",
        "/FillStation/LiteralNone": "nOnE",
    }
    source_classes = {vial: "vial", ROOT: None, MESH: "vial", **nullish}
    prim_types = {path: "Xform" for path in source_classes}
    prim_types[MESH] = "Mesh"
    existing = {path: () for path in source_classes}

    plan = plan_semantic_assignments(source_classes, prim_types, existing)

    assert {(item.path, item.class_name) for item in plan.assignments} == {
        (vial, "vial"),
        (ROOT, "liquid"),
    }
    assert plan.excluded_subtree == (ROOT, MESH)


@pytest.mark.parametrize(
    ("prim_types", "expected_reason"),
    [
        ({MESH: "Mesh"}, "liquid_root_missing"),
        ({ROOT: "Xform"}, "liquid_mesh_missing"),
        ({ROOT: "Xform", MESH: "Xform"}, "liquid_mesh_wrong_type"),
    ],
)
def test_plan_rejects_missing_or_wrong_reviewed_prims(prim_types, expected_reason):
    with pytest.raises(SemanticPlanError) as caught:
        plan_semantic_assignments({}, prim_types, {})

    assert caught.value.reason_code == expected_reason


@pytest.mark.parametrize("conflict_path", [ROOT, MESH, MESH + "/Nested"])
def test_plan_rejects_existing_class_taxonomy_anywhere_in_liquid_subtree(
    conflict_path,
):
    prim_types = {ROOT: "Xform", MESH: "Mesh", MESH + "/Nested": "Xform"}
    existing = {path: () for path in prim_types}
    existing[conflict_path] = ("vial",)

    with pytest.raises(SemanticPlanError) as caught:
        plan_semantic_assignments({}, prim_types, existing)

    assert caught.value.reason_code == "liquid_class_conflict"


def test_class_ids_resolve_from_integer_string_and_inherited_labels():
    resolved = resolve_class_ids(
        {
            3: {"class": "vial"},
            "8": {"class": "container, liquid"},
            "9": {"class": "background"},
        }
    )

    assert resolved.vial_ids == (3,)
    assert resolved.liquid_ids == (8,)
    assert resolved.records[1].labels == ("container", "liquid")


@pytest.mark.parametrize(
    "mapping",
    [
        {"not-decimal": {"class": "liquid"}},
        {"1.0": {"class": "liquid"}},
        {True: {"class": "liquid"}},
        {1: {"class": "liquid"}, "1": {"class": "vial"}},
        {"1": {"class": None}},
        {"1": {"class": "liquid,,container"}},
    ],
)
def test_class_id_resolution_rejects_malformed_or_colliding_mapping(mapping):
    with pytest.raises(LabelMappingError):
        resolve_class_ids(mapping)


def test_exact_semantic_class_mask_is_union_of_every_liquid_class_id():
    semantic = np.array(
        [[0, 2, 3], [4, 3, 0]],
        dtype=np.uint32,
    )
    mapping = {
        "0": {"class": "BACKGROUND"},
        "2": {"class": "liquid"},
        "3": {"class": "container, liquid"},
        "4": {"class": "vial"},
    }

    result = render_capture.exact_semantic_class_mask(
        semantic,
        mapping,
        class_name="liquid",
    )

    assert result.class_name == "liquid"
    assert result.semantic_ids == (2, 3)
    assert result.visible_semantic_ids == (0, 2, 3, 4)
    assert result.id_to_labels == (
        (0, ("BACKGROUND",)),
        (2, ("liquid",)),
        (3, ("container", "liquid")),
        (4, ("vial",)),
    )
    assert result.pixel_counts_by_id == ((2, 1), (3, 2))
    assert result.pixel_count == 3
    np.testing.assert_array_equal(
        result.mask,
        np.array([[0, 255, 255], [0, 255, 0]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("semantic", "mapping", "reason"),
    [
        (
            np.array([[0, 2]], dtype=np.float32),
            {0: {"class": "BACKGROUND"}, 2: {"class": "liquid"}},
            "semantic_dtype_invalid",
        ),
        (
            np.array([[0, 2]], dtype=np.uint32),
            {0: {"class": "BACKGROUND"}},
            "semantic_id_unmapped",
        ),
        (
            np.array([[0, 1]], dtype=np.uint32),
            {0: {"class": "BACKGROUND"}, 1: {"class": "vial"}},
            "semantic_class_missing",
        ),
        (
            np.array([[0, 1]], dtype=np.uint32),
            {
                0: {"class": "BACKGROUND"},
                1: {"class": "vial"},
                2: {"class": "liquid"},
            },
            "semantic_class_pixels_empty",
        ),
        (
            np.array([[2, 2]], dtype=np.uint32),
            {2: {"class": "liquid"}},
            "semantic_mask_whole_frame",
        ),
        (
            np.array([[0, 2]], dtype=np.uint32),
            {
                0: {"class": "BACKGROUND"},
                2: {"class": "liquid"},
                "2": {"class": "vial"},
            },
            "semantic_id_collision",
        ),
    ],
)
def test_exact_semantic_class_mask_fails_closed(
    semantic, mapping, reason
):
    with pytest.raises(LabelMappingError) as caught:
        render_capture.exact_semantic_class_mask(
            semantic,
            mapping,
            class_name="liquid",
        )

    assert caught.value.reason_code == reason


def test_exact_instance_path_mask_uses_only_reviewed_leaf_paths():
    instance = np.array(
        [[0, 11, 11, 12], [0, 13, 14, 12]], dtype=np.uint32
    )
    mapping = {
        "11": "/FillStation/FluidmixF2/ParticleSystem/Isosurface",
        "12": "/FillStation/FluidmixF2/RenderStreams/T200",
        "13": "/FillStation/FillStation_LocalRoot/Fill_vial_active/Mesh_998",
        "14": "/FillStation/Unreviewed",
    }
    resolved = resolve_instance_paths(mapping)

    result = exact_instance_path_mask(
        instance,
        mapping,
        allowed_paths=(
            "/FillStation/FluidmixF2/ParticleSystem/Isosurface",
            "/FillStation/FluidmixF2/RenderStreams/T200",
        ),
        required_paths=(
            "/FillStation/FluidmixF2/ParticleSystem/Isosurface",
        ),
    )

    assert resolved[0].path == "/FillStation/FluidmixF2/ParticleSystem/Isosurface"
    assert result.instance_ids == (11, 12)
    assert result.id_to_path == (
        (11, "/FillStation/FluidmixF2/ParticleSystem/Isosurface"),
        (12, "/FillStation/FluidmixF2/RenderStreams/T200"),
    )
    assert result.paths == (
        "/FillStation/FluidmixF2/ParticleSystem/Isosurface",
        "/FillStation/FluidmixF2/RenderStreams/T200",
    )
    assert result.pixel_counts_by_path == (
        ("/FillStation/FluidmixF2/ParticleSystem/Isosurface", 2),
        ("/FillStation/FluidmixF2/RenderStreams/T200", 2),
    )
    assert result.pixel_count == 4
    np.testing.assert_array_equal(
        result.mask,
        np.array([[0, 255, 255, 255], [0, 0, 0, 255]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    ("instance", "mapping", "required_paths", "reason"),
    [
        (
            np.array([[0, 11]], dtype=np.float32),
            {0: "BACKGROUND", 11: "/Liquid"},
            ("/Liquid",),
            "instance_dtype_invalid",
        ),
        (
            np.array([[0, 11]], dtype=np.uint32),
            {0: "BACKGROUND"},
            ("/Liquid",),
            "instance_id_unmapped",
        ),
        (
            np.array([[0, 11]], dtype=np.uint32),
            {0: "BACKGROUND", 11: "/Other"},
            ("/Liquid",),
            "required_instance_path_absent",
        ),
    ],
)
def test_exact_instance_path_mask_fails_closed(
    instance, mapping, required_paths, reason
):
    with pytest.raises(LabelMappingError) as caught:
        exact_instance_path_mask(
            instance,
            mapping,
            allowed_paths=("/Liquid",),
            required_paths=required_paths,
        )

    assert caught.value.reason_code == reason


def _valid_capture_inputs():
    rgb = np.array(
        [
            [[10, 20, 30], [11, 20, 30], [12, 20, 30]],
            [[13, 20, 30], [14, 20, 30], [15, 20, 30]],
        ],
        dtype=np.uint8,
    )
    semantic = np.array([[1, 1, 2], [2, 0, 0]], dtype=np.uint32)
    instance = np.array([[4, 4, 7], [7, 0, 0]], dtype=np.uint64)
    semantic_map = {
        "0": {"class": "BACKGROUND"},
        "1": {"class": "vial"},
        "2": {"class": "liquid"},
    }
    instance_map = {
        0: "BACKGROUND",
        4: "/FillStation/Vial",
        7: MESH,
    }
    return rgb, semantic, instance, semantic_map, instance_map


def test_valid_aligned_shared_capture_reports_exact_counts_and_ids():
    result = validate_capture(
        *_valid_capture_inputs(),
        raster_shape=(2, 3),
        require_liquid_pixels=True,
    )

    assert result.valid
    assert result.reason_codes == ()
    assert result.semantic_ids == (0, 1, 2)
    assert result.instance_ids == (0, 4, 7)
    assert result.class_mapping is not None
    assert result.class_mapping.liquid_ids == (2,)
    assert result.class_mapping.vial_ids == (1,)
    assert result.pixel_count_for("liquid") == 2
    assert result.pixel_count_for("vial") == 2
    assert result.selected_liquid_instance_ids == (7,)
    assert result.liquid_instance_pixel_count == 2
    assert result.instance_agreement


@pytest.mark.parametrize(
    ("mutate", "expected_reason"),
    [
        (lambda values: values.__setitem__(0, values[0][:, :2]), "rgb_shape_invalid"),
        (
            lambda values: values.__setitem__(1, values[1][:, :2]),
            "semantic_shape_invalid",
        ),
        (
            lambda values: values.__setitem__(2, values[2][:, :2]),
            "instance_shape_invalid",
        ),
        (
            lambda values: values.__setitem__(0, np.zeros((2, 3, 3), dtype=np.uint8)),
            "rgb_spatially_constant",
        ),
        (
            lambda values: values.__setitem__(0, values[0].astype(np.float32)),
            "rgb_dtype_invalid",
        ),
        (lambda values: values[3].pop("2"), "semantic_id_unmapped"),
        (lambda values: values[4].pop(7), "instance_id_unmapped"),
        (lambda values: values.__setitem__(4, {}), "instance_id_unmapped"),
        (
            lambda values: values[3].__setitem__("2", {"class": "vial,liquid"}),
            "class_ids_overlap",
        ),
        (
            lambda values: values[4].__setitem__(7, MESH + "/Unreviewed"),
            "liquid_instance_path_invalid",
        ),
        (
            lambda values: values[2].__setitem__((1, 2), 7),
            "liquid_instance_mismatch",
        ),
        (lambda values: values.__setitem__(0, values[0].tolist()), "rgb_not_array"),
        (
            lambda values: values[3].__setitem__("2", {"class": None}),
            "semantic_mapping_invalid",
        ),
        (lambda values: values[4].__setitem__(7, None), "instance_mapping_invalid"),
    ],
)
def test_capture_rejects_malformed_or_misaligned_inputs(mutate, expected_reason):
    values = list(_valid_capture_inputs())
    mutate(values)

    result = validate_capture(
        *values,
        raster_shape=(2, 3),
        require_liquid_pixels=True,
    )

    assert not result.valid
    assert expected_reason in result.reason_codes


@pytest.mark.parametrize(
    ("class_id", "instance_id", "expected_reason"),
    [
        (2, 7, "liquid_pixels_whole_frame"),
        (1, 4, "vial_pixels_whole_frame"),
    ],
)
def test_required_class_cannot_be_absent_or_fill_whole_frame(
    class_id, instance_id, expected_reason
):
    values = list(_valid_capture_inputs())
    values[1] = np.full((2, 3), class_id, dtype=np.uint32)
    values[2] = np.full((2, 3), instance_id, dtype=np.uint64)

    result = validate_capture(
        *values,
        raster_shape=(2, 3),
        require_liquid_pixels=True,
    )

    assert not result.valid
    assert expected_reason in result.reason_codes


def test_prefill_allows_zero_liquid_but_positive_smoke_does_not():
    values = list(_valid_capture_inputs())
    values[1] = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint32)
    values[2] = np.array([[4, 4, 0], [4, 0, 0]], dtype=np.uint64)

    prefill = validate_capture(
        *values,
        raster_shape=(2, 3),
        require_liquid_pixels=False,
    )
    positive = validate_capture(
        *values,
        raster_shape=(2, 3),
        require_liquid_pixels=True,
    )

    assert prefill.valid
    assert prefill.pixel_count_for("liquid") == 0
    assert not positive.valid
    assert "liquid_pixels_absent" in positive.reason_codes


def test_renderer_is_import_safe_and_help_safe_without_isaac_imports():
    module = _load_renderer_module()
    completed = subprocess.run(
        [sys.executable, str(RENDER_SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert hasattr(module, "main")
    assert completed.returncode == 0, completed.stderr
    assert "--require-liquid-pixels" in completed.stdout


def test_renderer_rejects_existing_and_dangling_output_paths(tmp_path):
    module = _load_renderer_module()
    existing_file = tmp_path / "existing"
    existing_file.write_text("occupied")
    existing_directory = tmp_path / "directory"
    existing_directory.mkdir()
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing-target")

    for path in (existing_file, existing_directory, dangling):
        with pytest.raises(FileExistsError):
            module.validate_output_destination(path)


@pytest.mark.parametrize("broken_annotator", ["semantic", "instance"])
def test_renderer_retries_after_transient_annotator_envelope_failure(
    broken_annotator,
):
    module = _load_renderer_module()
    rgb, semantic, instance, semantic_map, instance_map = _valid_capture_inputs()
    valid_semantic = {"data": semantic, "info": {"idToLabels": semantic_map}}
    valid_instance = {"data": instance, "info": {"idToLabels": instance_map}}
    semantic_results = iter(
        [np.zeros(0, dtype=np.uint32), valid_semantic]
        if broken_annotator == "semantic"
        else [valid_semantic, valid_semantic]
    )
    instance_results = iter(
        [{"data": instance}, valid_instance]
        if broken_annotator == "instance"
        else [valid_instance, valid_instance]
    )
    steps = []

    result = module._retry_shared_capture(
        step=lambda: steps.append("step"),
        get_rgb=lambda: rgb,
        get_semantic=lambda: next(semantic_results),
        get_instance=lambda: next(instance_results),
        np_module=np,
        max_tries=2,
        raster_shape=(2, 3),
        require_liquid_pixels=True,
    )

    assert result.attempts == 2
    assert result.validation.valid
    assert steps == ["step", "step"]


def test_renderer_retry_exhaustion_retains_final_reasons_and_denominator():
    module = _load_renderer_module()
    rgb, semantic, instance, semantic_map, instance_map = _valid_capture_inputs()
    rgb = np.zeros_like(rgb)
    steps = []

    with pytest.raises(RuntimeError) as caught:
        module._retry_shared_capture(
            step=lambda: steps.append("step"),
            get_rgb=lambda: rgb,
            get_semantic=lambda: {
                "data": semantic,
                "info": {"idToLabels": semantic_map},
            },
            get_instance=lambda: {
                "data": instance,
                "info": {"idToLabels": instance_map},
            },
            np_module=np,
            max_tries=2,
            raster_shape=(2, 3),
            require_liquid_pixels=True,
        )

    assert steps == ["step", "step"]
    assert type(caught.value) is module.CaptureRetryExhausted
    assert caught.value.attempt_denominator == 2
    assert caught.value.last_reason_codes == ("rgb_spatially_constant",)
    assert caught.value.last_validation_summary["rgb"] == {
        "shape": [2, 3, 3],
        "dtype": "uint8",
        "minimum": 0.0,
        "maximum": 0.0,
        "mean": 0.0,
        "std": 0.0,
    }
    assert caught.value.last_validation_summary["semantic_ids"] == [0, 1, 2]


def test_renderer_makes_transient_envelope_fatal_only_after_retry_budget():
    module = _load_renderer_module()
    rgb, _, instance, _, instance_map = _valid_capture_inputs()
    steps = []

    with pytest.raises(module.CaptureRetryExhausted) as caught:
        module._retry_shared_capture(
            step=lambda: steps.append("step"),
            get_rgb=lambda: rgb,
            get_semantic=lambda: {},
            get_instance=lambda: {
                "data": instance,
                "info": {"idToLabels": instance_map},
            },
            np_module=np,
            max_tries=2,
            raster_shape=(2, 3),
            require_liquid_pixels=True,
        )

    assert steps == ["step", "step"]
    assert caught.value.attempt_denominator == 2
    assert caught.value.last_reason_codes == (
        "semantic_annotator_envelope_invalid",
    )
    assert caught.value.last_validation_summary is None


def test_renderer_persists_and_flushes_safe_failure_before_immediate_kit_exit(
    tmp_path,
):
    module = _load_renderer_module()
    rgb, semantic, instance, semantic_map, instance_map = _valid_capture_inputs()
    rgb = np.full_like(rgb, 231)
    validation = validate_capture(
        rgb,
        semantic,
        instance,
        semantic_map,
        instance_map,
        raster_shape=(2, 3),
        require_liquid_pixels=True,
    )
    failure = module.CaptureRetryExhausted(
        attempt_denominator=20,
        last_reason_codes=validation.reason_codes,
        last_validation=validation,
    )
    partial = tmp_path / ".capture.partial"
    partial.mkdir()

    class ImmediateExit(BaseException):
        pass

    class RecordingStderr(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flush_count = 0

        def flush(self):
            self.flush_count += 1
            super().flush()

    stderr = RecordingStderr()

    class FakeKit:
        def __init__(self):
            self.close_calls = []

        def close(self, *, exit_code=None):
            self.close_calls.append(exit_code)
            assert (partial / "failure.json").is_file()
            assert stderr.flush_count > 0
            raise ImmediateExit

    kit = FakeKit()

    def fail_capture(context):
        context.phase = "frame_capture"
        context.requested_time_code = 686
        raise failure

    with pytest.raises(ImmediateExit):
        module._run_capture_lifecycle(
            kit=kit,
            partial=partial,
            output=tmp_path / "capture",
            expected_frames=1,
            capture_body=fail_capture,
            stderr=stderr,
        )

    record = json.loads((partial / "failure.json").read_text())
    stderr_record = json.loads(stderr.getvalue())
    assert kit.close_calls == [1]
    assert record["schema_version"] == 1
    assert record["mechanical_gates_passed"] is False
    assert record["phase"] == "frame_capture"
    assert record["requested_time_code"] == 686
    assert record["exception"]["type"] == "CaptureRetryExhausted"
    assert record["attempt_denominator"] == 20
    assert record["last_reason_codes"] == ["rgb_spatially_constant"]
    assert record["last_validation"]["rgb"] == {
        "shape": [2, 3, 3],
        "dtype": "uint8",
        "minimum": 231.0,
        "maximum": 231.0,
        "mean": 231.0,
        "std": 0.0,
    }
    assert stderr_record == {"event": "twin_render_failure", "failure": record}
    serialized = json.dumps(record, sort_keys=True)
    for forbidden in ("raw", "data"):
        assert forbidden not in serialized


def test_renderer_surfaces_failure_json_write_error_before_shutdown(
    tmp_path, monkeypatch
):
    module = _load_renderer_module()
    partial = tmp_path / ".capture.partial"
    partial.mkdir()
    stderr = io.StringIO()

    def fail_write(path, record):
        raise OSError("simulated durable write failure")

    monkeypatch.setattr(module, "_write_json_atomic_durable", fail_write)

    class FakeKit:
        def __init__(self):
            self.close_calls = []

        def close(self, *, exit_code=None):
            self.close_calls.append(exit_code)

    kit = FakeKit()
    with pytest.raises(module.FailureDiagnosticWriteError):
        with module._diagnostic_kit_lifecycle(
            kit=kit,
            partial=partial,
            stderr=stderr,
        ) as context:
            context.phase = "frame_capture"
            context.requested_time_code = 686
            raise RuntimeError("original capture failure")

    stderr_record = json.loads(stderr.getvalue())
    assert kit.close_calls == [1]
    assert stderr_record["failure"]["diagnostic_write"] == {
        "status": "failed",
        "exception_type": "OSError",
        "message": "failure.json could not be written durably",
    }
    assert stderr_record["failure"]["mechanical_gates_passed"] is False
    assert not (partial / "failure.json").exists()


def _complete_capture_fixture(module, tmp_path):
    partial = tmp_path / ".render.partial"
    artifact = partial / "rgb" / "0001.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"complete capture evidence")
    input_file = tmp_path / "input.usda"
    input_file.write_text("#usda 1.0\n")
    summary = {
        "schema_version": 1,
        "mechanical_gates_passed": True,
        "visual_check_status": "pending",
        "frames": [
            {
                "artifacts": {
                    "rgb": {
                        "path": "rgb/0001.png",
                        "sha256": module.sha256_file(artifact),
                    }
                }
            }
        ],
    }
    (partial / "summary.json").write_text(json.dumps(summary))
    run = module.CaptureRun(
        summary=summary,
        input_snapshots=(module.snapshot_input_file("stage", input_file),),
    )
    return partial, tmp_path / "render", run


def test_renderer_capture_entry_delegates_to_tested_production_orchestration(
    tmp_path,
    monkeypatch,
):
    module = _load_renderer_module()
    output = tmp_path / "render"
    partial = tmp_path / ".render.partial"
    partial.mkdir()
    inputs = module.Preflight(
        twin=tmp_path,
        stage=tmp_path / "stage.usda",
        camera_file=tmp_path / "FillCam.json",
        script=module.REPO_ROOT / "scripts" / "twin_render_sequence.py",
        output=output,
        camera_record={},
        resolution=(640, 480),
        frame_range=(1, 4, 2),
        time_codes=(1, 3),
        input_snapshots=(),
    )
    simulation_configs = []

    class FakeIsaacModule:
        @staticmethod
        def SimulationApp(config):
            simulation_configs.append(config)
            return object()

    monkeypatch.setitem(sys.modules, "isaacsim", FakeIsaacModule)
    sentinel = module.CaptureRun(summary={}, input_snapshots=())
    orchestration_calls = []

    def fake_orchestration(**kwargs):
        orchestration_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(module, "_run_capture_lifecycle", fake_orchestration)

    result = module._capture_to_partial(
        module.argparse.Namespace(renderer="PathTracing"),
        inputs,
        partial,
    )

    assert result is sentinel
    assert simulation_configs == [
        {
            "headless": True,
            "renderer": "PathTracing",
            "width": 640,
            "height": 480,
            "extra_args": [
                "--/exts/isaacsim.core.throttling/enable_async=false"
            ],
        }
    ]
    assert len(orchestration_calls) == 1
    assert orchestration_calls[0]["partial"] == partial
    assert orchestration_calls[0]["output"] == output
    assert orchestration_calls[0]["expected_frames"] == 2
    assert callable(orchestration_calls[0]["capture_body"])


def test_renderer_publishes_validated_capture_before_immediate_success_close(
    tmp_path,
):
    module = _load_renderer_module()
    partial, output, run = _complete_capture_fixture(module, tmp_path)
    capture_calls = []

    class FakeKit:
        def close(self, *, exit_code=None):
            assert exit_code is None
            assert not partial.exists()
            assert (output / "rgb" / "0001.png").read_bytes() == (
                b"complete capture evidence"
            )
            raise SystemExit(0)

    def capture_body(context):
        capture_calls.append("capture")
        return run

    with pytest.raises(SystemExit) as caught:
        module._run_capture_lifecycle(
            kit=FakeKit(),
            partial=partial,
            output=output,
            expected_frames=1,
            capture_body=capture_body,
        )

    assert caught.value.code == 0
    assert capture_calls == ["capture"]
    assert output.is_dir()


def test_renderer_returns_from_normal_close_only_after_success_publication(tmp_path):
    module = _load_renderer_module()
    partial, output, run = _complete_capture_fixture(module, tmp_path)

    class FakeKit:
        def __init__(self):
            self.close_calls = []

        def close(self, *, exit_code=None):
            self.close_calls.append(exit_code)

    kit = FakeKit()
    module._run_capture_lifecycle(
        kit=kit,
        partial=partial,
        output=output,
        expected_frames=1,
        capture_body=lambda context: run,
    )

    assert kit.close_calls == [None]
    assert output.is_dir()
    assert not partial.exists()
    assert not (output / "failure.json").exists()


@pytest.mark.parametrize(
    "shutdown_error",
    [SystemExit(7), type("PostPublicationShutdown", (BaseException,), {})()],
    ids=["nonzero-system-exit", "custom-base-exception"],
)
def test_renderer_propagates_postpublication_shutdown_without_diagnostic_or_reclose(
    tmp_path,
    shutdown_error,
):
    module = _load_renderer_module()
    partial, output, run = _complete_capture_fixture(module, tmp_path)

    class FakeKit:
        def __init__(self):
            self.close_calls = []

        def close(self, *, exit_code=None):
            self.close_calls.append(exit_code)
            partial.mkdir()
            raise shutdown_error

    kit = FakeKit()
    with pytest.raises(type(shutdown_error)) as caught:
        module._run_capture_lifecycle(
            kit=kit,
            partial=partial,
            output=output,
            expected_frames=1,
            capture_body=lambda context: run,
        )

    assert caught.value is shutdown_error
    assert kit.close_calls == [None]
    assert output.is_dir()
    assert partial.is_dir()
    assert not (partial / "failure.json").exists()
    assert not (output / "failure.json").exists()


def test_renderer_main_does_not_mislabel_published_shutdown_as_retained(
    tmp_path,
    monkeypatch,
    capsys,
):
    module = _load_renderer_module()
    partial = tmp_path / ".render.partial"
    partial.mkdir()
    output = tmp_path / "render"
    inputs = module.Preflight(
        twin=tmp_path,
        stage=tmp_path / "stage.usda",
        camera_file=tmp_path / "FillCam.json",
        script=module.REPO_ROOT / "scripts" / "twin_render_sequence.py",
        output=output,
        camera_record={},
        resolution=(3, 2),
        frame_range=(1, 2, 1),
        time_codes=(1,),
        input_snapshots=(),
    )
    monkeypatch.setattr(module, "preflight", lambda args: inputs)
    monkeypatch.setattr(module, "_create_partial_output", lambda path: partial)

    def publish_then_shutdown(args, actual_inputs, actual_partial):
        assert actual_inputs is inputs
        module.publish_partial_no_clobber(actual_partial, actual_inputs.output)
        actual_partial.mkdir()
        raise SystemExit(0)

    monkeypatch.setattr(module, "_capture_to_partial", publish_then_shutdown)

    with pytest.raises(SystemExit) as caught:
        module.main(["--twin", str(tmp_path), "--out", str(output)])

    assert caught.value.code == 0
    assert output.is_dir()
    assert partial.is_dir()
    assert "retained partial output" not in capsys.readouterr().err


@pytest.mark.parametrize("failure_kind", ["validation", "input_drift"])
def test_renderer_diagnoses_finalization_failure_before_nonzero_close(
    tmp_path,
    failure_kind,
):
    module = _load_renderer_module()
    partial, output, run = _complete_capture_fixture(module, tmp_path)
    if failure_kind == "validation":
        (partial / "rgb" / "0001.png").unlink()
        expected_error = RuntimeError
        expected_phase = "complete_partial_validation"
    else:
        run.input_snapshots[0].path.write_text("#usda 1.0\nchanged\n")
        expected_error = module.InputDriftError
        expected_phase = "final_input_reverification"
    stderr = io.StringIO()

    class FakeKit:
        def __init__(self):
            self.close_calls = []

        def close(self, *, exit_code=None):
            self.close_calls.append(exit_code)
            assert (partial / "failure.json").is_file()

    kit = FakeKit()
    with pytest.raises(expected_error):
        module._run_capture_lifecycle(
            kit=kit,
            partial=partial,
            output=output,
            expected_frames=1,
            capture_body=lambda context: run,
            stderr=stderr,
        )

    record = json.loads((partial / "failure.json").read_text())
    assert kit.close_calls == [1]
    assert record["phase"] == expected_phase
    assert record["exception"]["type"] == expected_error.__name__
    assert not output.exists()
    assert partial.is_dir()
    assert json.loads(stderr.getvalue()) == {
        "event": "twin_render_failure",
        "failure": record,
    }


def test_renderer_retains_finalization_failure_when_publication_loses_race(
    tmp_path,
):
    module = _load_renderer_module()
    partial, output, run = _complete_capture_fixture(module, tmp_path)
    output.mkdir()
    (output / "competitor.txt").write_text("competitor evidence")
    stderr = io.StringIO()

    class FakeKit:
        def __init__(self):
            self.close_calls = []

        def close(self, *, exit_code=None):
            self.close_calls.append(exit_code)
            assert (partial / "failure.json").is_file()

    kit = FakeKit()
    with pytest.raises(FileExistsError):
        module._run_capture_lifecycle(
            kit=kit,
            partial=partial,
            output=output,
            expected_frames=1,
            capture_body=lambda context: run,
            stderr=stderr,
        )

    record = json.loads((partial / "failure.json").read_text())
    assert kit.close_calls == [1]
    assert record["phase"] == "publication"
    assert record["exception"]["type"] == "FileExistsError"
    assert json.loads(stderr.getvalue()) == {
        "event": "twin_render_failure",
        "failure": record,
    }
    assert (output / "competitor.txt").read_text() == "competitor evidence"
    assert (partial / "rgb" / "0001.png").read_bytes() == (
        b"complete capture evidence"
    )


def test_renderer_publication_does_not_clobber_simulated_competitor(tmp_path):
    module = _load_renderer_module()
    partial = tmp_path / ".render.partial"
    partial.mkdir()
    (partial / "kept.txt").write_text("complete partial")
    output = tmp_path / "render"

    def competitor_then_atomic(source, destination):
        destination.mkdir()
        module._atomic_rename_noreplace(source, destination)

    with pytest.raises(FileExistsError):
        module.publish_partial_no_clobber(
            partial,
            output,
            rename_noreplace=competitor_then_atomic,
        )

    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert (partial / "kept.txt").read_text() == "complete partial"

    output.rmdir()
    module.publish_partial_no_clobber(partial, output)
    assert (output / "kept.txt").read_text() == "complete partial"
    assert not partial.exists()


def test_renderer_rejects_input_drift_between_snapshot_and_publication(tmp_path):
    module = _load_renderer_module()
    camera_file = tmp_path / "FillCam.json"
    camera_file.write_text('{"render_px": [640, 480]}')
    snapshots = (module.snapshot_input_file("camera", camera_file),)

    module.verify_input_snapshots(snapshots)
    camera_file.write_text('{"render_px": [320, 240]}')

    with pytest.raises(module.InputDriftError, match="camera"):
        module.verify_input_snapshots(snapshots)


def test_renderer_cli_has_validated_seed_frames_and_no_semantic_skip_route():
    module = _load_renderer_module()
    parser = module.build_parser()
    defaults = parser.parse_args(["--twin", "/tmp/twin", "--out", "/tmp/out"])

    assert defaults.seed == 20261024
    assert module.parse_frame_spec("1:7:2") == (1, 3, 5)
    assert module.parse_frame_spec("171,342,686") == (171, 342, 686)
    with pytest.raises(ValueError):
        module.parse_frame_spec("1:7:0")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--twin",
                "/tmp/twin",
                "--out",
                "/tmp/out",
                "--seed",
                "-1",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--twin",
                "/tmp/twin",
                "--out",
                "/tmp/out",
                "--skip",
                "semantic",
            ]
        )


@pytest.mark.parametrize(
    "frame_spec",
    (
        "",
        "171,",
        ",171",
        "171,,342",
        "171,171,686",
        "342,171,686",
        "171,not-an-integer,686",
        "0,171,342",
        "171,342,687",
    ),
)
def test_renderer_rejects_malformed_duplicate_unsorted_or_out_of_stage_frame_lists(
    frame_spec,
):
    module = _load_renderer_module()

    with pytest.raises(ValueError):
        module.parse_frame_spec(frame_spec)

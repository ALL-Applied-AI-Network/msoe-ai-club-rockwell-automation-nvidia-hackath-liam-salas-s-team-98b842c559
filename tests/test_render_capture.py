"""CPU contracts for fail-closed FillCam semantic capture."""

import os
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fluidmix.render_capture import (
    LIQUID_OVERRIDE,
    LabelMappingError,
    SemanticPlanError,
    plan_semantic_assignments,
    resolve_class_ids,
    validate_capture,
)


ROOT = LIQUID_OVERRIDE.root_path
MESH = LIQUID_OVERRIDE.mesh_path
RENDER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "twin_render_sequence.py"


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


def test_renderer_cli_has_validated_seed_frames_and_no_semantic_skip_route():
    module = _load_renderer_module()
    parser = module.build_parser()
    defaults = parser.parse_args(["--twin", "/tmp/twin", "--out", "/tmp/out"])

    assert defaults.seed == 20261024
    assert module.parse_frame_spec("1:7:2") == (1, 7, 2)
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

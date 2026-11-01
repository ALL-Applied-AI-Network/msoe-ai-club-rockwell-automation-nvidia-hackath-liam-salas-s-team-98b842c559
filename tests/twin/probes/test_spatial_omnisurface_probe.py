"""CPU contracts for the bounded Isaac spatial OmniSurface capability probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "spatial_omnisurface_capability_probe_script",
    ROOT / "cli/twin/probes" / "spatial_omnisurface_capability_probe.py",
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
probe_script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(probe_script)

from twin.probes.spatial_omnisurface_probe import (
    build_fixture_spatial_mapping,
    build_probe_plan,
    build_static_probe_fixture,
    measure_constant_parity,
    record_vertex_color_readback,
    validate_probe_receipt,
)


def test_plan_freezes_same_family_material_and_constant_parity_first():
    plan = build_probe_plan()

    assert plan["experiment"] == "spatial_omnisurface_capability_probe"
    assert plan["expected_isaac_sim_version"] == "6.2.0"
    assert plan["camera_path"] == "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
    assert plan["renderer"] == "PathTracing"
    assert plan["resolution"] == [640, 480]
    assert plan["warmup_updates"] == 60
    assert plan["rt_subframes"] == 64
    assert plan["max_bounces"] == 16
    assert plan["max_specular_and_transmission_bounces"] == 16
    assert plan["wall_limit_seconds"] == 900
    assert plan["wrapper_mdl_sha256"] == (
        "83c9811cee6e3c3d0b45e612069e03c845c0bc827e92bc14bdfbe4f707be4ced"
    )
    assert plan["phase_order"] == [
        "original_omnisurface_constant",
        "wrapper_omnisurface_constant",
        "wrapper_omnisurface_spatial",
    ]
    assert plan["constant_parity_thresholds"] == {
        "mean_abs_rgb_max": 2.0,
        "p95_abs_rgb_max": 8.0,
        "fraction_abs_rgb_gt_16_max": 0.01,
    }
    assert plan["omnisurface_parameters"] == {
        "specular_transmission_color": (0.18, 0.55, 0.75),
        "specular_transmission_scattering_depth_m": 0.01,
        "thin_walled": False,
        "specular_transmission_weight": 1.0,
        "diffuse_reflection_weight": 0.0,
        "specular_reflection_ior": 1.333,
        "specular_reflection_roughness": 0.08,
        "enable_opacity": False,
    }
    assert plan["display_color_interpolation"] == "vertex"
    assert plan["probe_geometry"] == "static_closed_standard_usd_mesh"
    assert plan["generated_isosurface_tested"] is False
    assert plan["semantic_mask_tested"] is False
    assert plan["runtime_edit_target"] == "session_layer"
    assert plan["output_use"] == "capability_evidence_only_not_training_data"
    assert plan["claims"] == {
        "uncalibrated_surface_visualization": True,
        "volumetric_concentration": False,
        "physical_validation": False,
    }
    assert [row["boundary"] for row in plan["documented_support"]] == [
        "runtime_mdl_search_path_registration",
        "named_color_scene_data_lookup",
        "vertex_interpolated_display_color",
        "omnisurface_finite_transmission",
    ]
    assert plan["to_be_observed"] == [
        "wrapper_mdl_compiles_and_resolves_in_isaac_sim_6_2",
        "wrapper_constant_color_matches_original_omnisurface_within_frozen_thresholds",
        "vertex_display_color_changes_the_wrapper_render_spatially",
    ]


def test_constant_parity_uses_frozen_channel_metrics_and_thresholds():
    original = np.zeros((2, 3, 3), dtype=np.uint8)
    wrapper = np.ones((2, 3, 3), dtype=np.uint8)

    metrics, difference = measure_constant_parity(original, wrapper)

    assert metrics == {
        "mean_abs_rgb": 1.0,
        "p95_abs_rgb": 1.0,
        "fraction_abs_rgb_gt_16": 0.0,
        "passed": True,
    }
    assert difference.dtype == np.uint8
    np.testing.assert_array_equal(difference, np.ones_like(original))

    failed, _ = measure_constant_parity(
        original, np.full_like(original, 9, dtype=np.uint8)
    )
    assert failed["passed"] is False


@pytest.mark.parametrize(
    ("original", "wrapper", "message"),
    [
        (np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), "HxWx3"),
        (np.zeros((0, 2, 3), dtype=np.uint8), np.zeros((0, 2, 3), dtype=np.uint8), "nonempty"),
        (np.zeros((2, 2, 3), dtype=float), np.zeros((2, 2, 3), dtype=float), "integer"),
        (np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((3, 2, 3), dtype=np.uint8), "same shape"),
    ],
)
def test_constant_parity_fails_closed_on_malformed_rgb(original, wrapper, message):
    with pytest.raises((TypeError, ValueError), match=message):
        measure_constant_parity(original, wrapper)


def test_static_fixture_is_closed_vertex_interpolated_and_maps_multiple_compositions():
    fixture = build_static_probe_fixture()
    points = fixture["mesh_vertices_m"]
    counts = fixture["face_vertex_counts"]
    indices = fixture["face_vertex_indices"]

    assert points.shape == (18, 3)
    assert counts.shape == (24,)
    assert int(counts.sum()) == indices.shape[0]
    assert int(indices.min()) == 0
    assert int(indices.max()) == points.shape[0] - 1
    assert np.all(np.isfinite(points))
    assert fixture["display_color_interpolation"] == "vertex"

    colors, nearest_ids, evidence = build_fixture_spatial_mapping(points)
    assert colors.shape == points.shape
    assert nearest_ids.shape == (points.shape[0],)
    assert np.unique(nearest_ids).size >= 3
    assert np.unique(colors, axis=0).shape[0] >= 3
    assert evidence["mesh_vertex_count"] == points.shape[0]
    assert evidence["calibrated"] is False
    assert evidence["volumetric_concentration_claim"] is False

    again = build_fixture_spatial_mapping(points.copy())
    np.testing.assert_array_equal(again[0], colors)
    np.testing.assert_array_equal(again[1], nearest_ids)
    assert again[2] == evidence


def test_vertex_color_readback_requires_exact_unindexed_float32_values():
    authored = np.asarray(
        [[0.95, 0.38, 0.08], [0.18, 0.55, 0.75]], dtype=np.float64
    )
    readback = np.asarray(authored, dtype=np.float32)

    evidence = record_vertex_color_readback(
        authored,
        readback,
        interpolation="vertex",
        point_count=2,
        indices=None,
    )

    canonical = np.asarray(authored, dtype="<f4")
    import hashlib

    expected_hash = hashlib.sha256(canonical.tobytes()).hexdigest()
    assert evidence == {
        "interpolation": "vertex",
        "indexed": False,
        "point_count": 2,
        "value_count": 2,
        "exact_float32_readback": True,
        "authored_float32_sha256": expected_hash,
        "readback_float32_sha256": expected_hash,
        "readback_values": canonical.astype(float).tolist(),
    }


@pytest.mark.parametrize(
    ("readback", "interpolation", "point_count", "indices", "message"),
    [
        ([[0.95, 0.38, 0.08]], "vertex", 2, None, "point count"),
        ([[0.95, 0.38, 0.08], [0.18, 0.55, 0.74]], "vertex", 2, None, "differs"),
        ([[0.95, 0.38, 0.08], [0.18, 0.55, 0.75]], "faceVarying", 2, None, "vertex"),
        ([[0.95, 0.38, 0.08], [0.18, 0.55, 0.75]], "vertex", 2, [0, 1], "unindexed"),
    ],
)
def test_vertex_color_readback_fails_closed_on_topology_or_value_mismatch(
    readback, interpolation, point_count, indices, message
):
    authored = [[0.95, 0.38, 0.08], [0.18, 0.55, 0.75]]
    with pytest.raises(ValueError, match=message):
        record_vertex_color_readback(
            authored,
            readback,
            interpolation=interpolation,
            point_count=point_count,
            indices=indices,
        )


def passing_receipt():
    hash_a = "a" * 64
    hash_b = "b" * 64
    return {
        "schema_version": 1,
        "experiment": "spatial_omnisurface_capability_probe",
        "status": "pending_visual_review",
        "plan": build_probe_plan(),
        "arguments": {
            "stage": "/twin/stations/fill_station/fill_station.usda",
            "mdl": "/repo/assets/mdl/FluidmixSpatialOmniSurface.mdl",
            "seed": 20261025,
        },
        "runtime": {
            "isaac_sim_version": "6.2.0",
            "wall_seconds": 100.0,
            "wall_limit_seconds": 900,
        },
        "scene_contract": {
            "default_prim": "/FillStation",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "frozen_time_code": 1.0,
            "session_layer_only": True,
            "source_layers_unchanged": True,
            "camera_path": "/FillStation/FillStation_LocalRoot/FillCam/FillCam",
            "custom_camera_count": 0,
            "custom_light_count": 0,
            "custom_backdrop_count": 0,
        },
        "render": {
            "renderer": "PathTracing",
            "resolution": [640, 480],
            "warmup_updates": 60,
            "rt_subframes": 64,
            "max_bounces": 16,
            "max_specular_and_transmission_bounces": 16,
            "async_throttling_enabled": False,
            "original_constant_path": "original-constant.png",
            "original_constant_sha256": hash_a,
            "wrapper_constant_path": "wrapper-constant.png",
            "wrapper_constant_sha256": hash_b,
            "parity_absdiff_path": "constant-parity-absdiff.png",
            "parity_absdiff_sha256": hash_a,
            "wrapper_spatial_path": "wrapper-spatial.png",
            "wrapper_spatial_sha256": hash_b,
            "all_images_readable": True,
            "all_images_nonconstant": True,
        },
        "material": {
            "wrapper_mdl_sha256": build_probe_plan()["wrapper_mdl_sha256"],
            "mdl_search_path_registered": "/repo/assets/mdl",
            "wrapper_subidentifier": "FluidmixSpatialOmniSurface",
            "original_omnisurface_resolved": True,
            "wrapper_omnisurface_resolved": True,
            "original_bound": True,
            "wrapper_bound_constant": True,
            "wrapper_bound_spatial": True,
            "same_family_omnisurface_delegate": True,
            "only_transmission_color_source_differs": True,
            "o1b_parameters": build_probe_plan()["omnisurface_parameters"],
        },
        "constant_parity": {
            "thresholds": build_probe_plan()["constant_parity_thresholds"],
            "mean_abs_rgb": 1.0,
            "p95_abs_rgb": 2.0,
            "fraction_abs_rgb_gt_16": 0.001,
            "passed": True,
            "same_mesh_points_sha256": hash_a,
            "same_mesh_topology_sha256": hash_b,
            "same_display_color_sha256": hash_a,
            "same_time_code": True,
        },
        "spatial": {
            "mapping_evidence": {
                "calibrated": False,
                "physical_accuracy_claim": False,
                "volumetric_concentration_claim": False,
                "all_vertex_mappings_resolved": True,
                "vertex_colors_sha256": hash_a,
                "nearest_particle_ids_sha256": hash_b,
            },
            "primvar_readback": {
                "interpolation": "vertex",
                "indexed": False,
                "point_count": 18,
                "value_count": 18,
                "exact_float32_readback": True,
                "authored_float32_sha256": hash_a,
                "readback_float32_sha256": hash_a,
                "readback_values": [[0.1, 0.2, 0.3]] * 18,
            },
        },
        "claims": build_probe_plan()["claims"],
        "visual_review": {"status": "pending_top_level_review"},
        "errors": [],
        "warnings": [],
    }


def test_receipt_accepts_only_technical_pass_pending_visual_review():
    receipt = passing_receipt()
    assert validate_probe_receipt(receipt) == ()


@pytest.mark.parametrize(
    ("path", "value", "failure"),
    [
        (("material", "wrapper_omnisurface_resolved"), False, "mdl_resolution"),
        (("material", "wrapper_bound_spatial"), False, "material_binding"),
        (("constant_parity", "passed"), False, "constant_parity"),
        (("spatial", "primvar_readback", "exact_float32_readback"), False, "primvar_readback"),
        (("claims", "volumetric_concentration"), True, "claim_boundary"),
        (("status",), "pass", "status_pending_visual"),
    ],
)
def test_receipt_fails_closed_when_required_capability_evidence_is_missing(
    path, value, failure
):
    receipt = deepcopy(passing_receipt())
    target = receipt
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    assert failure in validate_probe_receipt(receipt)


def test_runner_parser_exposes_only_frozen_inputs_and_validates_before_kit(tmp_path):
    stage = tmp_path / "fill_station.usda"
    mdl = tmp_path / "FluidmixSpatialOmniSurface.mdl"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    mdl.write_bytes(
        (ROOT / "assets" / "mdl" / "FluidmixSpatialOmniSurface.mdl").read_bytes()
    )
    output = tmp_path / "probe"

    args = probe_script.build_parser().parse_args(
        ["--stage", str(stage), "--mdl", str(mdl), "--out", str(output)]
    )
    probe_script.validate_args(args)

    assert args.seed == 20261025
    assert vars(args) == {
        "stage": stage,
        "mdl": mdl,
        "out": output,
        "seed": 20261025,
    }
    assert "isaacsim" not in sys.modules


def test_runner_rejects_wrong_mdl_name_existing_output_and_nonabsolute_inputs(tmp_path):
    stage = tmp_path / "fill_station.usda"
    wrong_mdl = tmp_path / "wrong.mdl"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    wrong_mdl.write_text("mdl 1.7;\n", encoding="utf-8")
    output = tmp_path / "probe"

    args = probe_script.build_parser().parse_args(
        ["--stage", str(stage), "--mdl", str(wrong_mdl), "--out", str(output)]
    )
    with pytest.raises(ValueError, match="FluidmixSpatialOmniSurface.mdl"):
        probe_script.validate_args(args)

    output.mkdir()
    args.mdl = tmp_path / "FluidmixSpatialOmniSurface.mdl"
    args.mdl.write_text("mdl 1.7;\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output"):
        probe_script.validate_args(args)

    args.out = Path("relative-output")
    with pytest.raises(ValueError, match="absolute"):
        probe_script.validate_args(args)


def test_runner_rejects_unreviewed_wrapper_bytes_before_kit(tmp_path):
    stage = tmp_path / "fill_station.usda"
    mdl = tmp_path / "FluidmixSpatialOmniSurface.mdl"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    mdl.write_text("mdl 1.7;\n", encoding="utf-8")
    args = probe_script.build_parser().parse_args(
        [
            "--stage",
            str(stage),
            "--mdl",
            str(mdl),
            "--out",
            str(tmp_path / "probe"),
        ]
    )

    with pytest.raises(ValueError, match="reviewed asset hash"):
        probe_script.validate_args(args)


def test_runner_empty_receipt_is_explicitly_nonphysical_and_fail_closed(tmp_path):
    args = probe_script.build_parser().parse_args(
        [
            "--stage",
            str(tmp_path / "fill_station.usda"),
            "--mdl",
            str(tmp_path / "FluidmixSpatialOmniSurface.mdl"),
            "--out",
            str(tmp_path / "probe"),
        ]
    )

    receipt = probe_script.build_empty_receipt(args)

    assert receipt["status"] == "insufficient_evidence"
    assert receipt["plan"] == build_probe_plan()
    assert receipt["claims"] == {
        "uncalibrated_surface_visualization": True,
        "volumetric_concentration": False,
        "physical_validation": False,
    }
    assert receipt["material"]["original_omnisurface_resolved"] is False
    assert receipt["material"]["wrapper_omnisurface_resolved"] is False
    assert receipt["constant_parity"]["passed"] is False
    assert receipt["spatial"]["primvar_readback"] is None
    assert receipt["visual_review"]["status"] == "not_reached"


def test_runner_source_uses_runtime_mdl_resolution_vertex_primvar_and_no_a11_runner_import():
    source = (ROOT / "cli/twin/probes" / "spatial_omnisurface_capability_probe.py").read_text(
        encoding="utf-8"
    )

    assert "is_shader_resolved" in source
    assert "get_neuraylib" in source
    assert "addMdlSearchPath" in source
    assert "removeMdlSearchPath" in source
    assert 'CreatePrimvar("displayColor"' in source
    assert "UsdGeom.Tokens.vertex" in source
    assert "measure_constant_parity" in source
    assert "if not parity_metrics[\"passed\"]" in source
    assert "build_fixture_spatial_mapping" in source
    assert "record_vertex_color_readback" in source
    assert "physx_three_pour_sequence" not in source
    assert "SimulationApp" in source
    assert "WALL_LIMIT_SECONDS" in source
    assert '_progress("before_isaac_startup")' in source
    assert '_progress("isaac_startup_complete")' in source
    assert '_progress("wrapper_spatial_captured")' in source

"""Pure contract for a bounded spatial OmniSurface renderer capability probe."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
import re
from typing import Any

import numpy as np

from .physx_three_pour import (
    CAMERA_PATH,
    O1B_MATERIAL_PARAMETERS,
    PARITY_THRESHOLDS,
    SOURCE_LIQUID_BOTTOM_M,
    VIAL_CENTER_XY_M,
)
from .spatial_surface_color import build_surface_vertex_colors


EXPECTED_ISAAC_SIM_VERSION = "6.2.0"
EXPECTED_WRAPPER_MDL_SHA256 = (
    "83c9811cee6e3c3d0b45e612069e03c845c0bc827e92bc14bdfbe4f707be4ced"
)
PROBE_EXPERIMENT = "spatial_omnisurface_capability_probe"
WALL_LIMIT_SECONDS = 900
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_probe_receipt(receipt: object) -> tuple[str, ...]:
    """Return fail-closed technical capability receipt failures."""

    if not isinstance(receipt, Mapping):
        return ("receipt_mapping",)
    failures: list[str] = []

    def section(name: str) -> Mapping[str, object]:
        value = receipt.get(name)
        if isinstance(value, Mapping):
            return value
        failures.append(f"{name}_mapping")
        return {}

    def valid_hash(value: object) -> bool:
        return isinstance(value, str) and _SHA256.fullmatch(value) is not None

    if (
        receipt.get("schema_version") != 1
        or receipt.get("experiment") != PROBE_EXPERIMENT
        or receipt.get("plan") != build_probe_plan()
    ):
        failures.append("receipt_identity")
    if receipt.get("status") != "pending_visual_review":
        failures.append("status_pending_visual")

    runtime = section("runtime")
    observed_version = runtime.get("isaac_sim_version")
    version_suffix = (
        observed_version.removeprefix(EXPECTED_ISAAC_SIM_VERSION + ".")
        if isinstance(observed_version, str)
        else ""
    )
    wall_seconds = runtime.get("wall_seconds")
    if (
        observed_version != EXPECTED_ISAAC_SIM_VERSION
        and not (version_suffix and version_suffix.isdigit())
    ):
        failures.append("isaac_sim_version")
    if (
        isinstance(wall_seconds, bool)
        or not isinstance(wall_seconds, (int, float))
        or not math.isfinite(float(wall_seconds))
        or not 0.0 <= float(wall_seconds) <= WALL_LIMIT_SECONDS
        or runtime.get("wall_limit_seconds") != WALL_LIMIT_SECONDS
    ):
        failures.append("wall_limit")

    scene = section("scene_contract")
    if (
        scene.get("default_prim") != "/FillStation"
        or scene.get("meters_per_unit") != 1.0
        or scene.get("up_axis") != "Z"
        or scene.get("frozen_time_code") != 1.0
        or scene.get("session_layer_only") is not True
        or scene.get("source_layers_unchanged") is not True
        or scene.get("camera_path") != CAMERA_PATH
        or scene.get("custom_camera_count") != 0
        or scene.get("custom_light_count") != 0
        or scene.get("custom_backdrop_count") != 0
    ):
        failures.append("scene_contract")

    render = section("render")
    if (
        render.get("renderer") != "PathTracing"
        or render.get("resolution") != [640, 480]
        or render.get("warmup_updates") != 60
        or render.get("rt_subframes") != 64
        or render.get("max_bounces") != 16
        or render.get("max_specular_and_transmission_bounces") != 16
        or render.get("async_throttling_enabled") is not False
        or render.get("all_images_readable") is not True
        or render.get("all_images_nonconstant") is not True
        or any(
            not valid_hash(render.get(key))
            for key in (
                "original_constant_sha256",
                "wrapper_constant_sha256",
                "parity_absdiff_sha256",
                "wrapper_spatial_sha256",
            )
        )
    ):
        failures.append("render_contract")

    material = section("material")
    if (
        material.get("original_omnisurface_resolved") is not True
        or material.get("wrapper_omnisurface_resolved") is not True
        or not isinstance(material.get("mdl_search_path_registered"), str)
        or not material.get("mdl_search_path_registered")
        or material.get("wrapper_mdl_sha256") != EXPECTED_WRAPPER_MDL_SHA256
        or material.get("wrapper_subidentifier")
        != "FluidmixSpatialOmniSurface"
        or material.get("same_family_omnisurface_delegate") is not True
        or material.get("only_transmission_color_source_differs") is not True
        or material.get("o1b_parameters") != build_probe_plan()["omnisurface_parameters"]
    ):
        failures.append("mdl_resolution")
    if (
        material.get("original_bound") is not True
        or material.get("wrapper_bound_constant") is not True
        or material.get("wrapper_bound_spatial") is not True
    ):
        failures.append("material_binding")

    parity = section("constant_parity")
    parity_numbers: list[float] = []
    for key in (
        "mean_abs_rgb",
        "p95_abs_rgb",
        "fraction_abs_rgb_gt_16",
    ):
        value = parity.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            parity_numbers.append(float("inf"))
        else:
            parity_numbers.append(float(value))
    if (
        parity.get("thresholds") != build_probe_plan()["constant_parity_thresholds"]
        or not all(math.isfinite(value) and value >= 0.0 for value in parity_numbers)
        or parity_numbers[0] > PARITY_THRESHOLDS["mean_abs_rgb_max"]
        or parity_numbers[1] > PARITY_THRESHOLDS["p95_abs_rgb_max"]
        or parity_numbers[2] > PARITY_THRESHOLDS["fraction_abs_rgb_gt_16_max"]
        or parity.get("passed") is not True
        or parity.get("same_time_code") is not True
        or any(
            not valid_hash(parity.get(key))
            for key in (
                "same_mesh_points_sha256",
                "same_mesh_topology_sha256",
                "same_display_color_sha256",
            )
        )
    ):
        failures.append("constant_parity")

    spatial = section("spatial")
    mapping = spatial.get("mapping_evidence")
    mapping = mapping if isinstance(mapping, Mapping) else {}
    if (
        mapping.get("calibrated") is not False
        or mapping.get("physical_accuracy_claim") is not False
        or mapping.get("volumetric_concentration_claim") is not False
        or mapping.get("all_vertex_mappings_resolved") is not True
        or not valid_hash(mapping.get("vertex_colors_sha256"))
        or not valid_hash(mapping.get("nearest_particle_ids_sha256"))
    ):
        failures.append("spatial_mapping")
    readback = spatial.get("primvar_readback")
    readback = readback if isinstance(readback, Mapping) else {}
    if (
        readback.get("interpolation") != "vertex"
        or readback.get("indexed") is not False
        or readback.get("exact_float32_readback") is not True
        or readback.get("point_count") != readback.get("value_count")
        or not isinstance(readback.get("point_count"), int)
        or readback.get("point_count", 0) <= 0
        or not valid_hash(readback.get("authored_float32_sha256"))
        or readback.get("authored_float32_sha256")
        != readback.get("readback_float32_sha256")
        or not isinstance(readback.get("readback_values"), list)
        or len(readback.get("readback_values", [])) != readback.get("point_count")
    ):
        failures.append("primvar_readback")

    if receipt.get("claims") != build_probe_plan()["claims"]:
        failures.append("claim_boundary")
    visual = section("visual_review")
    if visual.get("status") != "pending_top_level_review":
        failures.append("visual_review_pending")
    if receipt.get("errors") != []:
        failures.append("errors_present")
    return tuple(dict.fromkeys(failures))


def build_static_probe_fixture() -> dict[str, Any]:
    """Return deterministic static mesh and particle fixture arrays."""

    segment_count = 8
    radius_m = 0.0105
    bottom_m = SOURCE_LIQUID_BOTTOM_M + 0.018
    top_m = bottom_m + 0.105
    rings: list[tuple[float, float, float]] = []
    for z in (bottom_m, top_m):
        rings.extend(
            (
                VIAL_CENTER_XY_M[0]
                + radius_m * math.cos(2.0 * math.pi * index / segment_count),
                VIAL_CENTER_XY_M[1]
                + radius_m * math.sin(2.0 * math.pi * index / segment_count),
                z,
            )
            for index in range(segment_count)
        )
    rings.extend(
        (
            VIAL_CENTER_XY_M[0],
            VIAL_CENTER_XY_M[1],
            z,
        )
        for z in (bottom_m, top_m)
    )
    face_counts: list[int] = []
    face_indices: list[int] = []
    for index in range(segment_count):
        following = (index + 1) % segment_count
        face_counts.append(4)
        face_indices.extend((index, following, following + segment_count, index + segment_count))
    bottom_center = 2 * segment_count
    top_center = bottom_center + 1
    for index in range(segment_count):
        following = (index + 1) % segment_count
        face_counts.append(3)
        face_indices.extend((bottom_center, following, index))
    for index in range(segment_count):
        following = (index + 1) % segment_count
        face_counts.append(3)
        face_indices.extend((top_center, index + segment_count, following + segment_count))

    particle_ids = np.asarray((40, 10, 70, 20, 90, 30), dtype=np.int64)
    low = bottom_m + 0.020
    high = top_m - 0.020
    offset = 0.0065
    particle_positions = np.asarray(
        (
            (VIAL_CENTER_XY_M[0] - offset, VIAL_CENTER_XY_M[1], low),
            (VIAL_CENTER_XY_M[0] + offset, VIAL_CENTER_XY_M[1], low),
            (VIAL_CENTER_XY_M[0], VIAL_CENTER_XY_M[1] - offset, (low + high) / 2.0),
            (VIAL_CENTER_XY_M[0], VIAL_CENTER_XY_M[1] + offset, (low + high) / 2.0),
            (VIAL_CENTER_XY_M[0] - offset, VIAL_CENTER_XY_M[1], high),
            (VIAL_CENTER_XY_M[0] + offset, VIAL_CENTER_XY_M[1], high),
        ),
        dtype=np.float64,
    )
    particle_compositions = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ),
        dtype=np.float64,
    )
    return {
        "mesh_vertices_m": np.asarray(rings, dtype=np.float64),
        "face_vertex_counts": np.asarray(face_counts, dtype=np.int32),
        "face_vertex_indices": np.asarray(face_indices, dtype=np.int32),
        "particle_ids": particle_ids,
        "particle_positions_m": particle_positions,
        "particle_compositions": particle_compositions,
        "display_color_interpolation": "vertex",
    }


def build_fixture_spatial_mapping(
    mesh_vertices_m: object,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map the stable fixture particles onto supplied mesh vertices."""

    fixture = build_static_probe_fixture()
    return build_surface_vertex_colors(
        fixture["particle_ids"],
        fixture["particle_positions_m"],
        fixture["particle_compositions"],
        mesh_vertices_m,
    )


def record_vertex_color_readback(
    authored_colors: object,
    readback_colors: object,
    *,
    interpolation: object,
    point_count: object,
    indices: object,
) -> dict[str, Any]:
    """Validate and hash exact USD Color3f vertex-primvar readback."""

    if interpolation != "vertex":
        raise ValueError("displayColor interpolation must be vertex")
    if isinstance(point_count, bool) or not isinstance(point_count, (int, np.integer)):
        raise TypeError("mesh point count must be an integer")
    expected_count = int(point_count)
    if expected_count <= 0:
        raise ValueError("mesh point count must be positive")
    if indices is not None:
        try:
            has_indices = len(indices) > 0
        except TypeError as exc:
            raise TypeError("displayColor indices must be an array or None") from exc
        if has_indices:
            raise ValueError("displayColor must be unindexed for exact vertex readback")

    def color_matrix(value: object, role: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.dtype.kind not in "fiu":
            raise TypeError(f"{role} must contain real color channels")
        result = np.asarray(raw, dtype=np.float64)
        if result.ndim != 2 or result.shape[1:] != (3,):
            raise ValueError(f"{role} must have shape (N, 3)")
        if result.shape[0] == 0 or not np.all(np.isfinite(result)):
            raise ValueError(f"{role} must be nonempty and finite")
        if np.any(result < 0.0) or np.any(result > 1.0):
            raise ValueError(f"{role} must be in [0, 1]")
        return result

    authored = color_matrix(authored_colors, "authored colors")
    readback = color_matrix(readback_colors, "readback colors")
    if authored.shape[0] != expected_count or readback.shape[0] != expected_count:
        raise ValueError("displayColor value count must equal mesh point count")
    authored_f32 = np.asarray(authored, dtype="<f4", order="C")
    readback_f32 = np.asarray(readback, dtype="<f4", order="C")
    if not np.array_equal(authored_f32, readback_f32):
        raise ValueError("displayColor readback differs from authored float32 values")
    authored_hash = hashlib.sha256(authored_f32.tobytes()).hexdigest()
    readback_hash = hashlib.sha256(readback_f32.tobytes()).hexdigest()
    return {
        "interpolation": "vertex",
        "indexed": False,
        "point_count": expected_count,
        "value_count": int(readback_f32.shape[0]),
        "exact_float32_readback": True,
        "authored_float32_sha256": authored_hash,
        "readback_float32_sha256": readback_hash,
        "readback_values": readback_f32.astype(float).tolist(),
    }


def measure_constant_parity(
    original_rgb: object, wrapper_rgb: object
) -> tuple[dict[str, Any], np.ndarray]:
    """Measure frozen full-frame RGB parity for the same static state."""

    original = np.asarray(original_rgb)
    wrapper = np.asarray(wrapper_rgb)
    for name, image in (("original_rgb", original), ("wrapper_rgb", wrapper)):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} must have shape HxWx3")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"{name} must be nonempty")
        if image.dtype.kind not in "iu":
            raise TypeError(f"{name} must contain integer RGB channels")
        if np.any(image < 0) or np.any(image > 255):
            raise ValueError(f"{name} channels must be in [0, 255]")
    if original.shape != wrapper.shape:
        raise ValueError("constant-parity RGB captures must have the same shape")
    difference = np.abs(
        original.astype(np.int16, copy=False) - wrapper.astype(np.int16, copy=False)
    ).astype(np.uint8)
    mean_abs = float(np.mean(difference))
    p95_abs = float(np.percentile(difference, 95))
    fraction_gt_16 = float(np.mean(difference > 16))
    metrics = {
        "mean_abs_rgb": mean_abs,
        "p95_abs_rgb": p95_abs,
        "fraction_abs_rgb_gt_16": fraction_gt_16,
        "passed": bool(
            mean_abs <= PARITY_THRESHOLDS["mean_abs_rgb_max"]
            and p95_abs <= PARITY_THRESHOLDS["p95_abs_rgb_max"]
            and fraction_gt_16
            <= PARITY_THRESHOLDS["fraction_abs_rgb_gt_16_max"]
        ),
    }
    return metrics, difference


def build_probe_plan() -> dict[str, object]:
    """Return the frozen capability-probe plan."""

    return {
        "experiment": PROBE_EXPERIMENT,
        "expected_isaac_sim_version": EXPECTED_ISAAC_SIM_VERSION,
        "camera_path": CAMERA_PATH,
        "renderer": "PathTracing",
        "resolution": [640, 480],
        "warmup_updates": 60,
        "rt_subframes": 64,
        "max_bounces": 16,
        "max_specular_and_transmission_bounces": 16,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "wrapper_mdl_sha256": EXPECTED_WRAPPER_MDL_SHA256,
        "phase_order": [
            "original_omnisurface_constant",
            "wrapper_omnisurface_constant",
            "wrapper_omnisurface_spatial",
        ],
        "constant_parity_thresholds": deepcopy(PARITY_THRESHOLDS),
        "omnisurface_parameters": deepcopy(O1B_MATERIAL_PARAMETERS),
        "display_color_interpolation": "vertex",
        "probe_geometry": "static_closed_standard_usd_mesh",
        "generated_isosurface_tested": False,
        "semantic_mask_tested": False,
        "runtime_edit_target": "session_layer",
        "output_use": "capability_evidence_only_not_training_data",
        "claims": {
            "uncalibrated_surface_visualization": True,
            "volumetric_concentration": False,
            "physical_validation": False,
        },
        "documented_support": [
            {
                "boundary": "runtime_mdl_search_path_registration",
                "status": "documented",
                "urls": [
                    "https://docs.omniverse.nvidia.com/kit/docs/omni.mdl.neuraylib/0.2.22/omni.mdl.neuraylib/omni.mdl.neuraylib.NeurayLib.html"
                ],
            },
            {
                "boundary": "named_color_scene_data_lookup",
                "status": "documented",
                "urls": [
                    "https://docs.omniverse.nvidia.com/extensions/latest/ext_material/ext_material-graph/nodes/Constants_State_Primvars/primvar_color.html",
                    "https://raytracing-docs.nvidia.com/mdl/specification/MDL_spec_1.10.2_14Mar2025.pdf",
                ],
            },
            {
                "boundary": "vertex_interpolated_display_color",
                "status": "documented",
                "urls": ["https://openusd.org/dev/user_guides/primvars.html"],
            },
            {
                "boundary": "omnisurface_finite_transmission",
                "status": "documented",
                "urls": [
                    "https://docs.omniverse.nvidia.com/materials-and-rendering/latest/templates/parameters/OmniSurface_Transmission.html"
                ],
            },
        ],
        "to_be_observed": [
            "wrapper_mdl_compiles_and_resolves_in_isaac_sim_6_2",
            "wrapper_constant_color_matches_original_omnisurface_within_frozen_thresholds",
            "vertex_display_color_changes_the_wrapper_render_spatially",
        ],
    }

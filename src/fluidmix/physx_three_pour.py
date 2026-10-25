"""Pure contracts for the bounded twin-based PhysX three-pour experiment."""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from .particle_diffusion import (
    DIFFUSION_DT_S as PARTICLE_DIFFUSION_DT_S,
    DIFFUSION_NEIGHBOR_RADIUS_M as PARTICLE_DIFFUSION_NEIGHBOR_RADIUS_M,
    EFFECTIVE_DIFFUSIVITY_M2_S as PARTICLE_EFFECTIVE_DIFFUSIVITY_M2_S,
)
from .particle_palette import (
    build_simplex_palette as build_particle_simplex_palette,
    mix_linear_rgb,
)


FLUID_REST_OFFSET_M = 0.0015
NOMINAL_PARTICLE_VOLUME_ML = 8.0 * FLUID_REST_OFFSET_M**3 * 1_000_000.0
PHYSICS_HZ = 60
CAPTURE_HZ = 10
CAPTURE_CADENCE_STEPS = PHYSICS_HZ // CAPTURE_HZ
PARTICLE_CONTACT_OFFSET_M = 0.003
DIFFUSION_NEIGHBOR_RADIUS_M = PARTICLE_DIFFUSION_NEIGHBOR_RADIUS_M
EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S = PARTICLE_EFFECTIVE_DIFFUSIVITY_M2_S
DIFFUSION_DT_S = PARTICLE_DIFFUSION_DT_S
DIFFUSION_STABILITY_CAP_PER_ROW = 0.25
COMPOSITION_COMPONENTS = ("t200", "t400", "t100")
COMPOSITION_PALETTE_DENOMINATOR = 4
F2_RECEIPT_SCHEMA_VERSION = 6
F2_MIXING_MODEL = "pbd_motion_composition_diffusion_global_blend_render_stream_fallback_v2"
F2_MIXING_ASSUMPTION = (
    "pbd_particle_motion_row_identity_diffusion_internal_global_render_fallback_v2"
)
PRIMARY_FINAL_CAPTURE_STEP = 2002
PRIMARY_FINAL_CAPTURE_OFFSET_S = 0.3
STAGE_GPU_MEMORY_GROWTH_STOP_MIB = 4096
TIMING_SOURCE_FIT = {
    "source": "video5_recipe_median_active_tank_linear_fit_v1",
    "source_fit_r2": 0.92595,
    "source_fit_rmse_s": 2.371,
    "source_fit_loocv_rmse_s": 4.765,
    "interpretation": "real_like_approximation_not_measured_tank_boundaries",
}
NOMINAL_TANK_TRANSMISSION_COLORS = {
    "t200": (0.95, 0.38, 0.08),
    "t400": (0.18, 0.55, 0.75),
    "t100": (0.92, 0.98, 1.0),
}
PALETTE_SMOKE_FALLBACK_TRIGGER = {
    "run_id": "20261025-brev-physx-palette-smoke-a1",
    "receipt_sha256": "c34da1875d098cc791ce2237d40cf4b32fcac14acab98f247dc32c98b9993e17",
    "visual_review_sha256": "8d8e1f0077340d9e64ce954bd32d365d6b6c4ca3252ac72b6977feb63190fe49",
    "decision": "fail",
    "failure_reasons": [
        "maximum_particle_movement_m_zero",
        "proto_indices_unchanged",
        "before_after_rgb_byte_identical",
    ],
}
CAMERA_PATH = "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
SOURCE_LIQUID_PATH = "/FillStation/FillStation_LocalRoot/Fill_liquid"
SOURCE_STREAM_PATH = "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream"
SOURCE_VIAL_PATH = "/FillStation/FillStation_LocalRoot/Fill_vial_active"
SOURCE_NOZZLE_PATH = "/FillStation/FillStation_LocalRoot/Fill_nozzle_stainless"
FLUIDMIX_NAMESPACE = "/FillStation/FluidmixF2"
TWIN_REFERENCE_TIME_CODE = 1.0
CAMERA_CLIPPING_RANGE = (0.0007619999814778566, 25.399999618530273)
VIAL_CENTER_XY_M = (5.2324003281, -1.9811999798)
SOURCE_LIQUID_RADIUS_M = 0.0132029216
SOURCE_LIQUID_BOTTOM_M = 0.9762065645
VIAL_MOUTH_M = 1.1762601692
NOZZLE_EXIT_CENTER_M = (5.23189233, -1.97916798, 1.17760319)
O1B_MATERIAL_PARAMETERS = {
    "specular_transmission_color": (0.18, 0.55, 0.75),
    "specular_transmission_scattering_depth_m": 0.01,
    "thin_walled": False,
    "specular_transmission_weight": 1.0,
    "diffuse_reflection_weight": 0.0,
    "specular_reflection_ior": 1.333,
    "specular_reflection_roughness": 0.08,
    "enable_opacity": False,
}
PARITY_THRESHOLDS = {
    "mean_abs_rgb_max": 2.0,
    "p95_abs_rgb_max": 8.0,
    "fraction_abs_rgb_gt_16_max": 0.01,
}
VISIBILITY_CHECK_TIME_CODES = (1, 12, 21, 367, 368, 586, 660, 661, 686)
_POUR_ROWS = (
    ("t200", 10, 370, 706),
    ("t400", 10, 370, 605),
    ("t100", 40, 1481, 673),
)
_SUPPORTED_COLLIDERS = {"Cube", "Sphere", "Capsule"}


def build_twin_collider_specs() -> tuple[dict[str, object], ...]:
    """Return fixed invisible box containment fitted to the source vial interior."""

    wall_count = 12
    wall_thickness = 0.001
    inner_apothem = SOURCE_LIQUID_RADIUS_M * math.cos(math.pi / wall_count)
    chord = 2.0 * SOURCE_LIQUID_RADIUS_M * math.sin(math.pi / wall_count)
    wall_height = VIAL_MOUTH_M - SOURCE_LIQUID_BOTTOM_M
    center_z = SOURCE_LIQUID_BOTTOM_M + wall_height / 2.0
    floor_thickness = 0.002
    values: list[dict[str, object]] = [
        {
            "name": "Floor",
            "path": f"{FLUIDMIX_NAMESPACE}/Colliders/Floor",
            "type": "Cube",
            "center_m": (VIAL_CENTER_XY_M[0], VIAL_CENTER_XY_M[1], SOURCE_LIQUID_BOTTOM_M - floor_thickness / 2.0),
            "size_m": (2.0 * SOURCE_LIQUID_RADIUS_M, 2.0 * SOURCE_LIQUID_RADIUS_M, floor_thickness),
            "rotate_z_degrees": 0.0,
            "visible": False,
        }
    ]
    for index in range(wall_count):
        angle = 2.0 * math.pi * index / wall_count
        radial_center = inner_apothem + wall_thickness / 2.0
        values.append(
            {
                "name": f"Wall{index:02d}",
                "path": f"{FLUIDMIX_NAMESPACE}/Colliders/Wall{index:02d}",
                "type": "Cube",
                "center_m": (VIAL_CENTER_XY_M[0] + radial_center * math.cos(angle), VIAL_CENTER_XY_M[1] + radial_center * math.sin(angle), center_z),
                "size_m": (wall_thickness, chord + 0.0005, wall_height),
                "rotate_z_degrees": math.degrees(angle),
                "visible": False,
            }
        )
    return tuple(values)


def _positive_real(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _composition_array(value: object, name: str = "compositions") -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numbers, not booleans or objects")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,):
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"each {name} row must sum to one")
    return array


def build_source_compositions(tank_id: str, particle_count: int) -> np.ndarray:
    """Return one-hot source compositions in T200/T400/T100 component order."""

    if tank_id not in COMPOSITION_COMPONENTS:
        raise ValueError(f"unknown source tank {tank_id!r}")
    if isinstance(particle_count, bool) or not isinstance(particle_count, int):
        raise TypeError("particle_count must be an integer")
    if particle_count <= 0:
        raise ValueError("particle_count must be positive")
    rows = np.zeros((particle_count, len(COMPOSITION_COMPONENTS)), dtype=np.float64)
    rows[:, COMPOSITION_COMPONENTS.index(tank_id)] = 1.0
    return rows


def build_composition_palette(
    denominator: int = COMPOSITION_PALETTE_DENOMINATOR,
) -> np.ndarray:
    """Return a deterministic finite lattice over the three-component simplex."""

    return build_particle_simplex_palette(denominator)


def build_palette_transmission_colors(compositions: object | None = None) -> np.ndarray:
    """Linearly map composition rows to uncalibrated transmission parameters."""

    rows = build_composition_palette() if compositions is None else _composition_array(compositions)
    tank_colors = np.asarray(
        [NOMINAL_TANK_TRANSMISSION_COLORS[tank] for tank in COMPOSITION_COMPONENTS],
        dtype=np.float64,
    )
    return mix_linear_rgb(rows, tank_colors)


def map_compositions_to_palette(
    compositions: object,
    palette: object | None = None,
) -> np.ndarray:
    """Map each composition to its nearest prototype; ties use first palette index."""

    rows = _composition_array(compositions)
    prototypes = build_composition_palette() if palette is None else _composition_array(palette, "palette")
    if prototypes.shape[0] == 0:
        raise ValueError("palette must not be empty")
    squared_distance = np.sum((rows[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    return np.argmin(squared_distance, axis=1).astype(np.int64, copy=False)


def build_global_blend_stage_states() -> tuple[dict[str, object], ...]:
    """Return the fallback's particle-weighted global vial colour at each stage."""

    cumulative_counts = np.zeros(len(COMPOSITION_COMPONENTS), dtype=np.int64)
    tank_colors = np.asarray(
        [NOMINAL_TANK_TRANSMISSION_COLORS[tank] for tank in COMPOSITION_COMPONENTS],
        dtype=np.float64,
    )
    states: list[dict[str, object]] = []
    cumulative_step = 0
    for ordinal, (tank_id, _volume_ml, particle_count, duration_steps) in enumerate(
        _POUR_ROWS, start=1
    ):
        cumulative_counts[COMPOSITION_COMPONENTS.index(tank_id)] += particle_count
        cumulative_step += duration_steps
        fractions = cumulative_counts.astype(np.float64) / float(cumulative_counts.sum())
        states.append(
            {
                "ordinal": ordinal,
                "tank_id": tank_id,
                "physics_step": (
                    PRIMARY_FINAL_CAPTURE_STEP
                    if ordinal == len(_POUR_ROWS)
                    else cumulative_step
                ),
                "cumulative_particle_counts": cumulative_counts.tolist(),
                "cumulative_particle_count": int(cumulative_counts.sum()),
                "component_fractions": fractions.tolist(),
                "transmission_color": (fractions @ tank_colors).tolist(),
            }
        )
    return tuple(states)


def _radius_neighbor_pairs(positions: np.ndarray, radius_m: float) -> list[tuple[int, int]]:
    cell_indices = np.floor(positions / radius_m).astype(np.int64)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for particle_index, cell in enumerate(cell_indices):
        cells.setdefault(tuple(int(value) for value in cell), []).append(particle_index)
    radius_squared = radius_m * radius_m
    pairs: list[tuple[int, int]] = []
    for particle_index, cell in enumerate(cell_indices):
        base = tuple(int(value) for value in cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other_index in cells.get(
                        (base[0] + dx, base[1] + dy, base[2] + dz), ()
                    ):
                        if other_index <= particle_index:
                            continue
                        delta = positions[other_index] - positions[particle_index]
                        if float(np.dot(delta, delta)) <= radius_squared:
                            pairs.append((particle_index, other_index))
    return pairs


def diffuse_particle_compositions(
    compositions: object,
    positions_m: object,
    *,
    diffusion_coefficient_m2_s: object = EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S,
    neighbor_radius_m: object = DIFFUSION_NEIGHBOR_RADIUS_M,
    dt_s: object = DIFFUSION_DT_S,
) -> np.ndarray:
    """Apply one conservative, stability-capped symmetric neighbour exchange."""

    rows = _composition_array(compositions)
    raw_positions = np.asarray(positions_m)
    if raw_positions.dtype.kind not in "fiu":
        raise TypeError("positions_m must contain real numbers, not booleans or objects")
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.shape != rows.shape:
        raise ValueError("positions_m must have shape (N, 3) matching compositions")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    coefficient = _positive_real(
        diffusion_coefficient_m2_s,
        "diffusion_coefficient_m2_s",
        allow_zero=True,
    )
    radius = _positive_real(neighbor_radius_m, "neighbor_radius_m")
    delta_time = _positive_real(dt_s, "dt_s")
    result = rows.copy()
    if rows.shape[0] < 2 or coefficient == 0.0:
        return result
    pairs = _radius_neighbor_pairs(positions, radius)
    if not pairs:
        return result
    degrees = np.zeros(rows.shape[0], dtype=np.int64)
    for first, second in pairs:
        degrees[first] += 1
        degrees[second] += 1
    exchange_weight = min(
        coefficient * delta_time / (radius * radius),
        DIFFUSION_STABILITY_CAP_PER_ROW / float(degrees.max()),
    )
    for first, second in pairs:
        exchange = exchange_weight * (rows[second] - rows[first])
        result[first] += exchange
        result[second] -= exchange
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise RuntimeError("diffusion violated finite non-negative composition")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("diffusion violated composition simplex row sums")
    if not np.allclose(result.sum(axis=0), rows.sum(axis=0), rtol=0.0, atol=1e-10):
        raise RuntimeError("diffusion violated global component conservation")
    return result


def _build_emission_schedule(
    particle_count: int,
    duration_steps: int,
    previous_end_step: int,
) -> tuple[dict[str, int], ...]:
    chunk_count = math.ceil(duration_steps / CAPTURE_CADENCE_STEPS)
    counts = distribute_particle_count(particle_count, chunk_count)
    return tuple(
        {
            "ordinal": index + 1,
            "physics_step": previous_end_step
            + math.ceil((index + 1) * duration_steps / chunk_count),
            "particle_count": count,
        }
        for index, count in enumerate(counts)
    )


def build_three_pour_plan() -> dict[str, object]:
    """Return the frozen F2 execution plan without importing Isaac Sim."""

    pours = []
    previous_end_step = 0
    for ordinal, (tank_id, volume_ml, particle_count, duration_steps) in enumerate(
        _POUR_ROWS, start=1
    ):
        equivalent = particle_count * NOMINAL_PARTICLE_VOLUME_ML
        schedule = _build_emission_schedule(
            particle_count,
            duration_steps,
            previous_end_step,
        )
        end_step = previous_end_step + duration_steps
        pours.append(
            {
                "ordinal": ordinal,
                "tank_id": tank_id,
                "programmed_volume_ml": volume_ml,
                "particle_count": particle_count,
                "start_step": previous_end_step + 1,
                "duration_steps": duration_steps,
                "duration_s": duration_steps / PHYSICS_HZ,
                "end_step": end_step,
                "stage_capture_step": (
                    PRIMARY_FINAL_CAPTURE_STEP if ordinal == len(_POUR_ROWS) else end_step
                ),
                "emission_chunk_count": len(schedule),
                "emission_schedule": list(schedule),
                "nominal_mass_convention_equivalent_ml": equivalent,
                "nominal_error_ml": equivalent - volume_ml,
            }
        )
        previous_end_step = end_step
    programmed = sum(row[1] for row in _POUR_ROWS)
    particle_count = sum(row[2] for row in _POUR_ROWS)
    nominal = particle_count * NOMINAL_PARTICLE_VOLUME_ML
    return {
        "schema_version": F2_RECEIPT_SCHEMA_VERSION,
        "experiment": "physx_three_pour_f2",
        "seed": 20261025,
        "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
        "nominal_particle_volume_ml": NOMINAL_PARTICLE_VOLUME_ML,
        "pours": pours,
        "totals": {"programmed_volume_ml": programmed, "particle_count": particle_count, "nominal_mass_convention_equivalent_ml": nominal, "nominal_error_ml": nominal - programmed},
        "timing": {
            "physics_hz": PHYSICS_HZ,
            "capture_hz": CAPTURE_HZ,
            "no_inter_pour_settle": True,
            "fill_end_step": previous_end_step,
            "primary_final_capture_step": PRIMARY_FINAL_CAPTURE_STEP,
            "primary_final_offset_s": PRIMARY_FINAL_CAPTURE_OFFSET_S,
            **TIMING_SOURCE_FIT,
        },
        "nominal_tank_transmission_colors": {
            tank: list(color) for tank, color in NOMINAL_TANK_TRANSMISSION_COLORS.items()
        },
        "transmission_colors_uncalibrated": True,
        "mixing_model": F2_MIXING_MODEL,
        "mixing_assumption": F2_MIXING_ASSUMPTION,
        "fallback_trigger": dict(PALETTE_SMOKE_FALLBACK_TRIGGER),
        "fallback_render_model": {
            "component_order": list(COMPOSITION_COMPONENTS),
            "physical_motion": "one_physx_pbd_vial_particle_system",
            "vial_representation": "visible_isosurface_one_global_material",
            "stream_representation": "render_only_mesh_current_tank_material",
            "global_blend_rule": "mean_composition_linear_light_rgb_uncalibrated",
            "blend_update_timing": "every_emission_event",
            "stage_states": list(build_global_blend_stage_states()),
            "spatial_mixing_color_rendered": False,
            "spatial_layering_color_rendered": False,
            "particle_composition_tracked": True,
            "diffusion_simulated": True,
            "composition_state": "internal_unrendered",
            "representation_limit": "one_global_vial_colour_cannot_render_layering_or_local_concentration",
        },
        "semantic_gate": {
            "required": True,
            "semantic_label": "liquid",
            "semantic_type": "class",
            "required_targets": ["visible_isosurface", "render_only_stream_mesh"],
            "annotator": "semantic_segmentation",
            "resolution": [640, 480],
            "aligned_mask_per_kept_rgb": True,
            "nonempty_liquid_pixels_per_mask": True,
            "unsupported_isosurface_semantics_outcome": "fail",
            "qualification_training_eligible": False,
        },
        "claims": {
            "physically_displaced_volume": False,
            "absorbance_fitted": False,
            "sim_to_real_validated": False,
            "particle_composition_tracked": True,
            "dye_concentration_simulated": False,
            "diffusion_simulated": True,
            "particle_composition_rendered": False,
            "spatial_mixing_color_rendered": False,
            "spatial_layering_color_rendered": False,
            "global_vial_blend_rendered": True,
            "source_stream_color_rendered": True,
            "mixing_physically_validated": False,
            "navier_stokes_validated": False,
        },
    }


def distribute_particle_count(total: int, chunks: int) -> tuple[int, ...]:
    """Distribute one pour's particles across nonempty ordered emission chunks."""

    for value, name in ((total, "total"), (chunks, "chunks")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if chunks > total:
        raise ValueError("chunks cannot exceed the particle total")
    quotient, remainder = divmod(total, chunks)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(chunks))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _append_once(failures: list[str], reason: str) -> None:
    if reason not in failures:
        failures.append(reason)


def validate_three_pour_receipt(receipt: Mapping[str, object]) -> tuple[str, ...]:
    """Return substantive twin-F2 failures; an empty tuple means the run passed."""

    if not isinstance(receipt, Mapping):
        return ("receipt_missing_or_unreadable",)
    failures: list[str] = []
    if receipt.get("schema_version") != F2_RECEIPT_SCHEMA_VERSION or receipt.get("experiment") != "physx_three_pour_f2":
        _append_once(failures, "receipt_identity")
    if receipt.get("status") != "pass":
        _append_once(failures, "run_status_not_pass")

    scene = _mapping(receipt.get("scene_contract"))
    layers = scene.get("source_layer_records")
    visibility = _mapping(scene.get("hidden_source_visibility"))
    visibility_ok = all(_mapping(visibility.get(path)).get(str(code)) == "invisible" for path in (SOURCE_LIQUID_PATH, SOURCE_STREAM_PATH) for code in VISIBILITY_CHECK_TIME_CODES)
    if (scene.get("default_prim") != "/FillStation" or scene.get("meters_per_unit") != 1.0 or scene.get("up_axis") != "Z" or scene.get("frozen_time_code") != TWIN_REFERENCE_TIME_CODE or scene.get("physics_namespace") != FLUIDMIX_NAMESPACE or scene.get("forbidden_authored_paths") != [] or scene.get("custom_camera_count") != 0 or scene.get("custom_light_count") != 0 or scene.get("custom_backdrop_count") != 0 or not visibility_ok or not isinstance(layers, list) or not layers or any(not _valid_hash(_mapping(row).get("sha256")) for row in layers) or scene.get("source_layers_unchanged") is not True):
        _append_once(failures, "twin_scene_contract")

    camera = _mapping(receipt.get("camera"))
    effective = camera.get("clipping_range_effective")
    if (camera.get("path") != CAMERA_PATH or camera.get("source_has_authored_clipping_range") is not True or camera.get("clipping_range_authored_by_runner") is not False or not isinstance(effective, list) or len(effective) != 2 or any(not _finite_number(value) for value in effective) or any(not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-9) for value, expected in zip(effective, CAMERA_CLIPPING_RANGE)) or camera.get("resolution") != [640, 480]):
        _append_once(failures, "source_fillcam_contract")

    material = _mapping(receipt.get("material"))
    expected_material_family = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in O1B_MATERIAL_PARAMETERS.items()
        if key != "specular_transmission_color"
    }
    expected_tank_colors = {
        tank: list(color) for tank, color in NOMINAL_TANK_TRANSMISSION_COLORS.items()
    }
    if (
        material.get("pbd_path") != f"{FLUIDMIX_NAMESPACE}/PBDMaterial"
        or material.get("vial_representation")
        != "visible_isosurface_one_global_material"
        or material.get("stream_representation")
        != "render_only_mesh_current_tank_material"
        or material.get("omnisurface_parameter_family") != expected_material_family
        or material.get("tank_transmission_colors") != expected_tank_colors
        or material.get("transmission_colors_uncalibrated") is not True
        or material.get("global_blend_rule")
        != "mean_composition_linear_light_rgb_uncalibrated"
        or material.get("isosurface_global_material_binding") is not True
    ):
        _append_once(failures, "fallback_material_contract")

    gpu = _mapping(receipt.get("gpu_simulation"))
    if gpu.get("cuda_device_available") is not True:
        _append_once(failures, "cuda_unavailable")
    if gpu.get("scene_enable_gpu_dynamics") is not True or gpu.get("broadphase") != "GPU":
        _append_once(failures, "gpu_dynamics_not_observed")

    system = _mapping(receipt.get("particle_system"))
    colliders = system.get("collider_types")
    collider_paths = system.get("collider_paths")
    if system.get("count") != 1 or system.get("exactly_one") is not True:
        _append_once(failures, "particle_system_count")
    if system.get("ccd_enabled") is not True:
        _append_once(failures, "particle_ccd_not_observed")
    if system.get("mass_api_count") != 0:
        _append_once(failures, "mass_api_present")
    if system.get("all_particle_sets_target_one_system") is not True:
        _append_once(failures, "particle_system_relation")
    if (not isinstance(colliders, list) or len(colliders) != len(build_twin_collider_specs()) or any(value not in _SUPPORTED_COLLIDERS for value in colliders) or not isinstance(collider_paths, list) or len(collider_paths) != len(build_twin_collider_specs()) or system.get("all_colliders_invisible") is not True or system.get("decorative_vial_collision_api") is not False):
        _append_once(failures, "twin_collider_contract")
    if system.get("pbd_material_binding") is not True or system.get("render_material_binding") is not True:
        _append_once(failures, "material_binding_missing")
    maximum_movement = system.get("maximum_particle_movement_m")
    if (
        system.get("finite_motion_observed") is not True
        or not _finite_number(maximum_movement)
        or float(maximum_movement) <= 0.0
    ):
        _append_once(failures, "particle_motion_not_observed")

    emitter = _mapping(receipt.get("emitter"))
    origin = emitter.get("origin_m")
    if (emitter.get("source_nozzle_path") != SOURCE_NOZZLE_PATH or emitter.get("source_nozzle_valid") is not True or emitter.get("velocity_m_s") != [0.0, 0.0, -0.1] or not isinstance(origin, list) or len(origin) != 3 or any(not _finite_number(value) or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-8) for value, expected in zip(origin, NOZZLE_EXIT_CENTER_M))):
        _append_once(failures, "nozzle_emitter_contract")

    parity = _mapping(receipt.get("pre_pour_parity"))
    if (parity.get("passed") is not True or parity.get("thresholds") != PARITY_THRESHOLDS or not _finite_number(parity.get("mean_abs_rgb")) or float(parity.get("mean_abs_rgb", math.inf)) > PARITY_THRESHOLDS["mean_abs_rgb_max"] or not _finite_number(parity.get("p95_abs_rgb")) or float(parity.get("p95_abs_rgb", math.inf)) > PARITY_THRESHOLDS["p95_abs_rgb_max"] or not _finite_number(parity.get("fraction_abs_rgb_gt_16")) or float(parity.get("fraction_abs_rgb_gt_16", math.inf)) > PARITY_THRESHOLDS["fraction_abs_rgb_gt_16_max"] or any(not _valid_hash(parity.get(key)) for key in ("reference_sha256", "f2_prefill_sha256", "absdiff_sha256"))):
        _append_once(failures, "pre_pour_twin_parity")

    raw_pours = receipt.get("pours")
    expected_cumulative = 0
    if not isinstance(raw_pours, list) or len(raw_pours) != len(_POUR_ROWS):
        _append_once(failures, "pour_order_or_volume")
        raw_pours = []
    plan_pours = build_three_pour_plan()["pours"]
    for ordinal, expected in enumerate(_POUR_ROWS, start=1):
        if ordinal > len(raw_pours):
            break
        row = _mapping(raw_pours[ordinal - 1])
        tank_id, volume_ml, particle_count, _ = expected
        plan_pour = plan_pours[ordinal - 1]
        expected_cumulative += particle_count
        if (row.get("ordinal") != ordinal or row.get("tank_id") != tank_id or row.get("programmed_volume_ml") != volume_ml or row.get("emitted_particle_count") != particle_count or row.get("cumulative_particle_count") != expected_cumulative):
            _append_once(failures, "pour_order_or_volume")
        if any(
            row.get(key) != plan_pour[key]
            for key in (
                "start_step",
                "duration_steps",
                "end_step",
                "stage_capture_step",
                "emission_chunk_count",
            )
        ):
            _append_once(failures, "real_timing_contract")
        if row.get("emission_events") != plan_pour["emission_schedule"]:
            _append_once(failures, "emission_schedule_mismatch")
        if row.get("finite_positions") is not True:
            _append_once(failures, "particle_output_nonfinite_or_missing")
        if (
            row.get("moved_under_physics") is not True
            or not _finite_number(row.get("maximum_particle_movement_m"))
            or float(row.get("maximum_particle_movement_m", 0.0)) <= 0.0
        ):
            _append_once(failures, "particle_motion_not_observed")
        if row.get("escaped_particle_count") != 0:
            _append_once(failures, "particles_escaped")
        if row.get("isosurface_nonempty") is not True:
            _append_once(failures, "isosurface_empty")
        before, after = row.get("used_gpu_memory_mib_before"), row.get("used_gpu_memory_mib_after")
        if not _finite_number(before) or not _finite_number(after):
            _append_once(failures, "memory_sample_missing")
        elif float(after) - float(before) >= STAGE_GPU_MEMORY_GROWTH_STOP_MIB:
            _append_once(failures, "stage_memory_growth_stop")

    totals = _mapping(receipt.get("totals"))
    plan_totals = build_three_pour_plan()["totals"]
    if (totals.get("programmed_volume_ml") != 60 or totals.get("emitted_particle_count") != 2221 or totals.get("final_particle_count") != 2221 or not _finite_number(totals.get("nominal_mass_convention_equivalent_ml")) or not math.isclose(float(totals.get("nominal_mass_convention_equivalent_ml", math.nan)), float(plan_totals["nominal_mass_convention_equivalent_ml"]), rel_tol=0.0, abs_tol=1e-9) or not _finite_number(totals.get("nominal_error_ml")) or not math.isclose(float(totals.get("nominal_error_ml", math.nan)), float(plan_totals["nominal_error_ml"]), rel_tol=0.0, abs_tol=1e-9)):
        _append_once(failures, "particle_or_nominal_volume_total")

    plan = build_three_pour_plan()
    if receipt.get("fallback_evidence") != plan["fallback_trigger"]:
        _append_once(failures, "fallback_trigger_mismatch")

    stream_render = _mapping(receipt.get("stream_render"))
    stream_records = stream_render.get("records")
    stream_paths: list[str] = []
    stream_hashes: list[str] = []
    stream_records_valid = (
        stream_render.get("representation")
        == "render_only_mesh_current_tank_material"
        and isinstance(stream_records, list)
        and len(stream_records) == len(plan_pours)
    )
    if stream_records_valid:
        for ordinal, (record, pour) in enumerate(zip(stream_records, plan_pours), start=1):
            row = _mapping(record)
            expected_color = list(NOMINAL_TANK_TRANSMISSION_COLORS[pour["tank_id"]])
            path = row.get("state_artifact_path")
            digest = row.get("state_artifact_sha256")
            if (
                row.get("ordinal") != ordinal
                or row.get("tank_id") != pour["tank_id"]
                or row.get("start_step") != pour["start_step"]
                or row.get("end_step") != pour["end_step"]
                or row.get("transmission_color") != expected_color
                or row.get("active_only_during_own_pour") is not True
                or not isinstance(path, str)
                or not path
                or not _valid_hash(digest)
            ):
                stream_records_valid = False
                break
            stream_paths.append(path)
            stream_hashes.append(digest)
    if (
        not stream_records_valid
        or len(set(stream_paths)) != len(plan_pours)
        or len(set(stream_hashes)) != len(plan_pours)
    ):
        _append_once(failures, "stream_render_contract")

    global_blend = _mapping(receipt.get("global_blend"))
    expected_blend = plan["fallback_render_model"]
    expected_update_count = sum(pour["emission_chunk_count"] for pour in plan_pours)
    if (
        global_blend.get("model") != plan["mixing_model"]
        or global_blend.get("component_order") != list(COMPOSITION_COMPONENTS)
        or global_blend.get("global_blend_rule") != expected_blend["global_blend_rule"]
        or global_blend.get("blend_update_timing") != "every_emission_event"
        or global_blend.get("updated_on_every_emission_event") is not True
        or global_blend.get("all_updates_usd_readback_verified") is not True
        or global_blend.get("particle_composition_tracked") is not True
        or global_blend.get("diffusion_simulated") is not True
        or global_blend.get("composition_state") != "internal_unrendered"
        or global_blend.get("spatial_mixing_color_rendered") is not False
        or global_blend.get("spatial_layering_color_rendered") is not False
    ):
        _append_once(failures, "global_blend_contract")
    if global_blend.get("blend_update_count") != expected_update_count:
        _append_once(failures, "global_blend_not_observed")
    if (
        global_blend.get("emission_trace_event_count") != expected_update_count
        or not isinstance(global_blend.get("emission_trace_path"), str)
        or not global_blend.get("emission_trace_path")
        or not _valid_hash(global_blend.get("emission_trace_sha256"))
    ):
        _append_once(failures, "global_blend_trace_missing")

    observed_states = global_blend.get("stage_states")
    expected_states = expected_blend["stage_states"]
    state_paths: list[str] = []
    state_hashes: list[str] = []
    state_values_valid = isinstance(observed_states, list) and len(observed_states) == len(expected_states)
    state_artifacts_valid = state_values_valid
    if state_values_valid:
        for observed, expected in zip(observed_states, expected_states):
            row = _mapping(observed)
            for key in (
                "ordinal",
                "tank_id",
                "physics_step",
                "cumulative_particle_counts",
                "cumulative_particle_count",
            ):
                if row.get(key) != expected[key]:
                    state_values_valid = False
            for key in ("component_fractions", "transmission_color"):
                value = row.get(key)
                if (
                    not isinstance(value, list)
                    or len(value) != 3
                    or any(not _finite_number(component) for component in value)
                    or not np.allclose(value, expected[key], rtol=0.0, atol=1e-12)
                ):
                    state_values_valid = False
            readback = row.get("usd_readback_transmission_color")
            if (
                not isinstance(readback, list)
                or len(readback) != 3
                or any(not _finite_number(component) for component in readback)
                or not np.allclose(
                    readback,
                    expected["transmission_color"],
                    rtol=0.0,
                    atol=1e-7,
                )
            ):
                state_values_valid = False
            path = row.get("state_artifact_path")
            digest = row.get("state_artifact_sha256")
            if not isinstance(path, str) or not path or not _valid_hash(digest):
                state_artifacts_valid = False
            else:
                state_paths.append(path)
                state_hashes.append(digest)
    if not state_values_valid:
        _append_once(failures, "global_blend_state_mismatch")
    if (
        not state_artifacts_valid
        or len(set(state_paths)) != len(expected_states)
        or len(set(state_hashes)) != len(expected_states)
    ):
        _append_once(failures, "global_blend_state_artifact_missing")

    composition_evidence = _mapping(receipt.get("composition_evidence"))
    expected_component_order = list(COMPOSITION_COMPONENTS)
    diffusion = _mapping(composition_evidence.get("diffusion"))
    if (
        composition_evidence.get("schema_version") != 1
        or composition_evidence.get("component_order") != expected_component_order
        or composition_evidence.get("internal_unrendered") is not True
        or diffusion.get("coefficient_m2_s") != EFFECTIVE_DIFFUSION_COEFFICIENT_M2_S
        or diffusion.get("neighbor_radius_m") != DIFFUSION_NEIGHBOR_RADIUS_M
        or diffusion.get("dt_s") != DIFFUSION_DT_S
        or not isinstance(composition_evidence.get("diffusion_update_count"), int)
        or composition_evidence.get("diffusion_update_count")
        != PRIMARY_FINAL_CAPTURE_STEP
    ):
        _append_once(failures, "composition_evidence_mismatch")
    evidence_stages = composition_evidence.get("stages")
    evidence_paths: list[str] = []
    evidence_hashes: list[str] = []
    evidence_valid = isinstance(evidence_stages, list) and len(evidence_stages) == len(plan_pours)
    if evidence_valid:
        for evidence, expected_pour in zip(evidence_stages, plan_pours):
            row = _mapping(evidence)
            expected_counts = [
                sum(
                    int(event["particle_count"])
                    for pour in plan_pours[: int(expected_pour["ordinal"])]
                    if pour["tank_id"] == component
                    for event in pour["emission_schedule"]
                )
                for component in COMPOSITION_COMPONENTS
            ]
            component_totals = row.get("component_totals")
            fractions = row.get("component_fractions")
            expected_fractions = [
                float(value) / float(sum(expected_counts)) for value in expected_counts
            ]
            checks_ok = (
                row.get("ordinal") == expected_pour["ordinal"]
                and row.get("tank_id") == expected_pour["tank_id"]
                and row.get("physics_step") == expected_pour["stage_capture_step"]
                and row.get("particle_count") == sum(expected_counts)
                and row.get("cumulative_particle_count")
                == sum(expected_counts)
                and isinstance(component_totals, list)
                and len(component_totals) == 3
                and all(
                    _finite_number(value)
                    and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1.0e-9)
                    for value, expected in zip(component_totals, expected_counts)
                )
                and isinstance(fractions, list)
                and len(fractions) == 3
                and all(
                    _finite_number(value)
                    and math.isclose(
                        float(value), expected, rel_tol=0.0, abs_tol=1.0e-12
                    )
                    for value, expected in zip(fractions, expected_fractions)
                )
                and _finite_number(row.get("row_sum_error"))
                and float(row.get("row_sum_error")) <= 1.0e-12
                and _finite_number(row.get("conservation_error"))
                and float(row.get("conservation_error")) <= 1.0e-9
                and row.get("nonnegative_passed") is True
                and row.get("conservation_passed") is True
                and row.get("particle_row_order_preserved") is True
                and row.get("diffusion_update_count") == expected_pour["stage_capture_step"]
                and isinstance(row.get("diffusion_neighbor_count_total"), int)
                and row.get("diffusion_neighbor_count_total") >= 0
                and isinstance(row.get("diffusion_max_degree"), int)
                and row.get("diffusion_max_degree") >= 0
                and _finite_number(row.get("diffusion_pair_weight_max"))
                and float(row.get("diffusion_pair_weight_max")) >= 0.0
                and isinstance(row.get("vertical_composition_summary"), list)
                and bool(row.get("vertical_composition_summary"))
                and all(_valid_hash(row.get(key)) for key in (
                    "particle_ids_sha256",
                    "positions_sha256",
                    "compositions_sha256",
                    "palette_indices_sha256",
                ))
            )
            path = row.get("artifact_path")
            digest = row.get("artifact_sha256")
            if (
                not checks_ok
                or not isinstance(path, str)
                or not path
                or not _valid_hash(digest)
            ):
                evidence_valid = False
            else:
                evidence_paths.append(path)
                evidence_hashes.append(digest)
    if (
        not evidence_valid
        or len(set(evidence_paths)) != len(plan_pours)
        or len(set(evidence_hashes)) != len(plan_pours)
    ):
        _append_once(failures, "composition_evidence_mismatch")

    isosurface = _mapping(receipt.get("isosurface"))
    if (isosurface.get("enabled") is not True or isosurface.get("visible_for_global_blend") is not True or not isinstance(isosurface.get("mesh_point_count"), int) or int(isosurface.get("mesh_point_count", 0)) <= 0 or not isinstance(isosurface.get("mesh_face_count"), int) or int(isosurface.get("mesh_face_count", 0)) <= 0 or isosurface.get("mesh_finite_points") is not True or isosurface.get("mesh_extent_nonempty") is not True):
        _append_once(failures, "isosurface_empty")

    render = _mapping(receipt.get("render"))
    frame_count = render.get("frame_count")
    frame_paths = render.get("frame_paths")
    frame_hashes = render.get("frame_sha256")
    frame_steps = render.get("frame_physics_steps")
    stage_indices = render.get("stage_frame_indices")
    stage_hashes = render.get("stage_frame_sha256")
    video_encoding_contract = (
        (
            render.get("video_codec"),
            render.get("video_container"),
            render.get("video_writer_fallback_used"),
        )
        in {("mp4v", "mp4", False), ("MJPG", "avi", True)}
    )
    if (not video_encoding_contract or render.get("renderer") != "PathTracing" or render.get("resolution") != [640, 480] or render.get("rt_subframes") != 64 or render.get("warmup_updates") != 60 or render.get("video_decodable") is not True or not isinstance(frame_count, int) or frame_count <= 0 or render.get("video_decoded_frame_count") != frame_count or not _valid_hash(render.get("video_sha256"))):
        _append_once(failures, "video_not_decodable")
    if (render.get("stage_frames_are_video_subsets") is not True or not isinstance(frame_paths, list) or len(frame_paths) != frame_count or any(not isinstance(path, str) or not path for path in frame_paths) or len(set(frame_paths)) != frame_count or not isinstance(frame_hashes, list) or len(frame_hashes) != frame_count or any(not _valid_hash(value) for value in frame_hashes) or not isinstance(stage_hashes, list) or len(stage_hashes) != 3 or not set(stage_hashes).issubset(set(frame_hashes))):
        _append_once(failures, "stage_frame_not_subset")
    expected_stage_steps = [pour["stage_capture_step"] for pour in plan_pours]
    expected_frame_steps = sorted(
        {
            *range(
                CAPTURE_CADENCE_STEPS,
                PRIMARY_FINAL_CAPTURE_STEP + 1,
                CAPTURE_CADENCE_STEPS,
            ),
            *expected_stage_steps,
        }
    )
    frame_sequence_valid = (
        frame_count == len(expected_frame_steps)
        and frame_steps == expected_frame_steps
        and isinstance(frame_paths, list)
        and len(frame_paths) == len(expected_frame_steps)
        and all(isinstance(path, str) and path for path in frame_paths)
        and len(set(frame_paths)) == len(expected_frame_steps)
        and isinstance(frame_hashes, list)
        and len(frame_hashes) == len(expected_frame_steps)
        and all(_valid_hash(value) for value in frame_hashes)
        and len(set(frame_hashes)) == len(expected_frame_steps)
    )
    if not frame_sequence_valid:
        _append_once(failures, "frame_sequence_contract")
    frame_timeline_valid = (
        frame_sequence_valid
        and all(isinstance(step, int) and not isinstance(step, bool) for step in frame_steps)
    )
    expected_stage_indices = (
        [frame_steps.index(step) for step in expected_stage_steps]
        if frame_timeline_valid
        else None
    )
    if (
        render.get("stage_frame_physics_steps") != expected_stage_steps
        or stage_indices != expected_stage_indices
        or not isinstance(stage_hashes, list)
        or len(stage_hashes) != len(expected_stage_steps)
        or len(set(stage_hashes)) != len(expected_stage_steps)
        or not frame_timeline_valid
        or not isinstance(frame_hashes, list)
        or expected_stage_indices is None
        or any(
            stage_hashes[index] != frame_hashes[frame_index]
            for index, frame_index in enumerate(expected_stage_indices or [])
        )
        or render.get("primary_final_frame_sha256")
        != (stage_hashes[-1] if stage_hashes else None)
        or not _valid_hash(render.get("primary_final_frame_sha256"))
    ):
        _append_once(failures, "primary_capture_contract")

    semantics = _mapping(receipt.get("semantics"))
    direct_isosurface = _mapping(semantics.get("isosurface_direct_binding"))
    if (
        semantics.get("semantic_label") != "liquid"
        or semantics.get("semantic_type") != "class"
        or semantics.get("isosurface_semantic_bound") is not True
        or semantics.get("isosurface_direct_binding_authored") is not True
        or not isinstance(semantics.get("isosurface_direct_binding_step"), int)
        or int(semantics.get("isosurface_direct_binding_step", 0)) <= 0
        or direct_isosurface.get("path")
        != f"{FLUIDMIX_NAMESPACE}/ParticleSystem/Isosurface"
        or direct_isosurface.get("schema") != "SemanticsLabelsAPI:class"
        or direct_isosurface.get("labels") != ["liquid"]
        or direct_isosurface.get("direct_binding_verified") is not True
        or semantics.get("stream_semantic_bound") is not True
    ):
        _append_once(failures, "liquid_semantic_binding_missing")
    semantic_pairs = semantics.get("pairs")
    semantic_pairs_valid = (
        semantics.get("annotator") == "semantic_segmentation"
        and semantics.get("resolution") == [640, 480]
        and semantics.get("semantic_gate_passed") is True
        and semantics.get("training_eligible") is False
        and semantics.get("ineligibility_reason")
        == "qualification_run_not_case_dataset"
        and isinstance(semantic_pairs, list)
        and isinstance(frame_count, int)
        and len(semantic_pairs) == frame_count
        and isinstance(frame_steps, list)
        and isinstance(frame_paths, list)
        and isinstance(frame_hashes, list)
    )
    mask_paths: list[str] = []
    mask_hashes: list[str] = []
    if semantic_pairs_valid:
        for index, raw_pair in enumerate(semantic_pairs):
            pair = _mapping(raw_pair)
            mask_path = pair.get("mask_path")
            mask_hash = pair.get("mask_sha256")
            liquid_pixels = pair.get("liquid_pixel_count")
            if (
                pair.get("physics_step") != frame_steps[index]
                or pair.get("rgb_path") != frame_paths[index]
                or pair.get("rgb_sha256") != frame_hashes[index]
                or not isinstance(mask_path, str)
                or not mask_path
                or not _valid_hash(mask_hash)
                or pair.get("mask_readable") is not True
                or pair.get("mask_resolution") != [640, 480]
                or not isinstance(liquid_pixels, int)
                or isinstance(liquid_pixels, bool)
                or liquid_pixels <= 0
            ):
                semantic_pairs_valid = False
                break
            mask_paths.append(mask_path)
            mask_hashes.append(mask_hash)
    semantic_stage_indices_valid = (
        isinstance(stage_indices, list)
        and len(stage_indices) == len(expected_stage_steps)
        and all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(mask_hashes)
            for index in stage_indices
        )
    )
    expected_stage_mask_hashes = (
        [mask_hashes[index] for index in stage_indices]
        if semantic_pairs_valid and semantic_stage_indices_valid
        else None
    )
    if (
        not semantic_pairs_valid
        or len(set(mask_paths)) != frame_count
        or semantics.get("stage_mask_sha256") != expected_stage_mask_hashes
        or not isinstance(expected_stage_mask_hashes, list)
        or len(set(expected_stage_mask_hashes)) != len(expected_stage_steps)
    ):
        _append_once(failures, "liquid_semantic_mask_invalid")

    memory = _mapping(receipt.get("memory"))
    memory_sample_contract = (
        (memory.get("sample_source"), memory.get("sample_scope"))
        in {
            ("nvidia_smi_gpu_memory_used", "gpu_framebuffer"),
            ("proc_meminfo_system_used_uma", "system_wide_unified_memory"),
        }
    )
    if (not memory_sample_contract or memory.get("hard_limit_mib") != 40000 or memory.get("hard_limit_exceeded") is not False or memory.get("stage_growth_stop_mib") != STAGE_GPU_MEMORY_GROWTH_STOP_MIB or memory.get("stage_growth_events") != [] or not _finite_number(memory.get("peak_used_mib")) or float(memory.get("peak_used_mib", math.inf)) >= 40000 or memory.get("isosurface_time_limit_seconds") != 2700 or not _finite_number(memory.get("isosurface_enabled_seconds")) or float(memory.get("isosurface_enabled_seconds", math.inf)) > 2700):
        _append_once(failures, "memory_limit_exceeded")
    timing = _mapping(receipt.get("timing"))
    if (timing.get("wall_limit_seconds") != 5400 or not _finite_number(timing.get("wall_seconds")) or float(timing.get("wall_seconds", math.inf)) > 5400):
        _append_once(failures, "wall_time_exceeded")
    plan_timing = build_three_pour_plan()["timing"]
    expected_timing = {
        "physics_hz": plan_timing["physics_hz"],
        "capture_hz": plan_timing["capture_hz"],
        "pour_duration_steps": [pour["duration_steps"] for pour in plan_pours],
        "pour_end_steps": [pour["end_step"] for pour in plan_pours],
        "no_inter_pour_settle": plan_timing["no_inter_pour_settle"],
        "fill_end_step": plan_timing["fill_end_step"],
        "primary_final_capture_step": plan_timing["primary_final_capture_step"],
        "primary_final_offset_s": plan_timing["primary_final_offset_s"],
        "executed_physics_steps": plan_timing["primary_final_capture_step"],
        "source": plan_timing["source"],
        "source_fit_r2": plan_timing["source_fit_r2"],
        "source_fit_rmse_s": plan_timing["source_fit_rmse_s"],
        "source_fit_loocv_rmse_s": plan_timing["source_fit_loocv_rmse_s"],
        "interpretation": plan_timing["interpretation"],
    }
    if any(timing.get(key) != expected for key, expected in expected_timing.items()):
        _append_once(failures, "real_timing_contract")
    plan = build_three_pour_plan()
    if (
        receipt.get("mixing_assumption") != plan["mixing_assumption"]
        or receipt.get("claims") != plan["claims"]
    ):
        _append_once(failures, "claim_limit_mismatch")
    errors = receipt.get("errors")
    if not isinstance(errors, list) or errors:
        _append_once(failures, "run_error_recorded")
    return tuple(failures)

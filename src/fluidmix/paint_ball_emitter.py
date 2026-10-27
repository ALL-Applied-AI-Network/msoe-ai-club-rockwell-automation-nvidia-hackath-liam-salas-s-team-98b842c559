"""CPU contracts for the NVIDIA Paint Ball Emitter F2 render route.

The Isaac-side runner owns the PointInstancer and PhysX attributes.  This
module keeps the append, prototype, containment and receipt rules testable
without importing Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .particle_palette import COMPOSITION_COMPONENTS
from .nuisance_authoring import (
    EXPECTED_LIGHT_PATHS,
    LIGHT_SPECS,
    apply_nuisance_session_layer,
    author_nuisance_session_layer,
    build_nuisance_authoring_plan,
    read_existing_light_inventory,
)
from .physx_particle_contract import (
    C21_RESOLUTION_PROFILE_ID,
    DENSITY_KG_M3,
    FLUID_REST_OFFSET_M,
    PARTICLE_CONTACT_OFFSET_M,
    PARTICLE_SET_MASS_OVERRIDE,
    NOMINAL_PARTICLE_VOLUME_ML,
    REST_OFFSET_M,
    SOLID_REST_OFFSET_M,
    SPAWN_SPACING_M,
    particle_resolution_profile,
    required_particle_capacity,
)
from .physx_dispense_residual import MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML

from .physx_volume_gate import (
    expected_cylindrical_height_m,
    find_dense_surface_row,
    height_within_tolerance,
    particle_bulk_height_from_quantiles_m,
    projected_top_dodecagon_min_y_px,
    projected_vial_geometry_roi_mask,
    visible_height_from_surface_row_m,
)
from .volume_height_curve import (
    height_bounds_for_volume_ml,
    height_for_volume_ml,
    volume_height_curve_digest,
)


PAINT_BALL_RECEIPT_SCHEMA_VERSION = 20
# NVIDIA's Replicator guidance calls for at least two real-time subframes when
# materials load late. The established FillCam paths use 32 or 64; 16 is the
# bounded lower-cost review candidate and still requires the visual gate.
SUPPORTED_TWIN_RT_SUBFRAMES = (16, 32, 64)
SUPPORTED_TWIN_PATHTRACING_TOTAL_SPP = (16, 32, 64)
PATHTRACING_SUBFRAME_DOCUMENTATION_URL = (
    "https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/"
    "subframes_examples.html"
)
SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL = (
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/"
    "dev_guide/schemas/physxschema.html"
)
MAX_VELOCITY_M_S = 2.5  # 250 cm/s in ParticlePostProcessingDemo.py at 0.01 m/unit.
FLOOR_CONTACT_OFFSET_M = 0.025
FLOOR_REST_OFFSET_M = 0.0
WALL_CONTACT_OFFSET_M = 0.025
WALL_REST_OFFSET_M = 0.0
FLOOR_COLLISION_DOCUMENTATION_URL = (
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/"
    "dev_guide/rigid_bodies_articulations/collision.html"
)
FLOOR_TUNNELING_FAILURE_RECEIPT_SHA256 = (
    "5ff739bf2fe49615c8fe174cb3400031a88cdc8bc2fbc851c92ab504b62547e3"
)
WALL_TUNNELING_FAILURE_RECEIPT_SHA256 = (
    "3ea78b1d91eb64b2dab30db7d0db9182d416d35abb851c2f6489d1ebf26e34da"
)
RENDER_PRODUCT_GATING_DOCUMENTATION_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/"
    "replicator_tutorials/tutorial_replicator_sdg_workflows.html"
)
RENDER_PRODUCT_CLEANUP_DOCUMENTATION_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/"
    "replicator_tutorials/tutorial_replicator_getting_started.html"
)
RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES = 48
SIMULATION_MANAGER_STEP_DOCUMENTATION_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/py/source/extensions/"
    "isaacsim.core.simulation_manager/docs/index.html#"
    "isaacsim.core.simulation_manager.SimulationManager.step"
)
TIMELINE_APP_UPDATE_DOCUMENTATION_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/"
    "replicator_tutorials/tutorial_replicator_getting_started.html#"
    "event-triggered-data-capture-timeline-and-simulation"
)
PERIODIC_CAPTURE_MEMORY_INTERVAL_FRAMES = 48
GPU_MEMORY_HARD_LIMIT_MIB = 40_000
F2_SAFE_CAPTURE_STRIDE_STEPS = 312
F2_VIDEO_PLAYBACK_FPS = 10.0
F2_WALL_LIMIT_SECONDS = 10_800
F2_MOTION_FALSIFIER_STEP = 12
OPAQUE_BASE_CUTOFF_VOLUME_ML = 20.0
OPAQUE_BASE_CUTOFF_HEIGHT_M = height_for_volume_ml(OPAQUE_BASE_CUTOFF_VOLUME_ML)
OPAQUE_BASE_CUTOFF_SOURCE = (
    "Liam operator note 2026-10-26: rough real-video-5 pixel reads place the "
    "opaque holder cutoff at approximately 15-20 mL; 20 mL freezes the "
    "conservative upper endpoint and is not a calibrated measurement"
)
EMITTER_RENDER_PATH = "point_instancer_sphere_emitter"
COUNT_DRIVEN_RENDER_PATH = "count_driven_twin_liquid_mesh"
PERSISTENT_RENDER_STREAM_PATH = "/FillStation/FluidmixF2/RenderStream"
COUNT_DRIVEN_DODECAGON_AREA_M2 = 0.0005229514163272396
# Historical count-driven rendering is retained only for receipt compatibility.
# The accepted particle runtime consumes Task 1's nominal volume directly via
# ``build_runtime_particle_configuration`` and does not use this retired body.
COUNT_DRIVEN_PARTICLE_VOLUME_M3 = 27.0e-9
COUNT_DRIVEN_SOURCE_GEOMETRY_SHA256 = (
    "e1b9711dc50195e559f54f96db2ce5389a34ef719213c443498a5585b6bd712f"
)
COUNT_DRIVEN_SOURCE_TOPOLOGY_SHA256 = (
    "edba8a108a846fbb3fd661e95b9a642b727a06e7ba431ca70e013d1255132acd"
)
TRANSPARENT_ANNOTATION_DOCUMENTATION_URL = (
    "https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/"
    "annotations_with_transparency.html"
)
PAIRED_CAPTURE_CAMERA_PATH = (
    "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
)
PAIRED_CAPTURE_STATE_FIELDS = (
    "callback_count",
    "timeline_time_s",
    "camera_path",
    "positions",
    "simulation_points",
    "velocities",
    "ids",
    "proto_indices",
)
RENDER_LIMITATIONS = {
    "continuous_surface": False,
    "palette_uncalibrated": True,
    "pbd_not_navier_stokes": True,
}
F2_REQUIRED_ACCEPTANCE_GATES = frozenset(
    {
        "scene_parity",
        "finite_particle_state",
        "containment",
        "zero_leakage",
        "volume_height",
        "stream_continuity",
        "source_color",
        "numeric_mixing",
        "visual_mixing",
        "semantics_alignment",
        "timing",
    }
)
COUNT_DRIVEN_RENDER_LIMITATIONS = {
    "continuous_surface": True,
    "palette_uncalibrated": True,
    "pbd_not_navier_stokes": True,
    "count_driven_geometry": True,
    "spatial_mixing_color_rendered": False,
    "spatial_layering_color_rendered": False,
}
_ARRAY_FIELDS = ("positions", "velocities", "orientations", "scales")
FULL_RUN_VOLUME_ROWS = ((1412, 10.0, 370), (2622, 20.0, 740), (4004, 60.0, 2221))
FULL_RUN_FRAME_STEPS = tuple(
    sorted({*range(12, 4005, 12), *(row[0] for row in FULL_RUN_VOLUME_ROWS)})
)
FULL_RUN_FRAME_COUNT = 336
FULL_RUN_STAGE_INDICES = tuple(
    FULL_RUN_FRAME_STEPS.index(row[0]) for row in FULL_RUN_VOLUME_ROWS
)
FULL_RUN_STREAM_ROWS = (
    (1, "t200", 1, 1412),
    (2, "t400", 1413, 2622),
    (3, "t100", 2623, 3968),
)


@dataclass(frozen=True, slots=True)
class SourceState:
    """Immutable identity shared by one pour's proxy stream and particle births."""

    scheduled_tank: str
    composition: tuple[float, float, float]
    palette_prototype_index: int
    material_path: str
    pour_event_id: str
    birth_event_id: str
    birth_step: int

    @property
    def stream_composition(self) -> tuple[float, float, float]:
        return self.composition

    @property
    def birth_composition(self) -> tuple[float, float, float]:
        return self.composition


def build_source_state(
    *,
    scheduled_tank: str,
    composition: Sequence[float],
    palette_prototype_index: int,
    material_path: str,
    pour_event_id: str,
    birth_event_id: str,
    birth_step: int,
) -> SourceState:
    """Validate and freeze the sole source identity for stream and emission."""

    if scheduled_tank not in COMPOSITION_COMPONENTS:
        raise ValueError(f"unknown scheduled tank {scheduled_tank!r}")
    raw = np.asarray(composition)
    if raw.dtype.kind not in "fiu":
        raise TypeError("source composition must contain real numbers")
    values = np.asarray(composition, dtype=np.float64)
    if (
        values.shape != (len(COMPOSITION_COMPONENTS),)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise ValueError(
            "source composition must be a finite non-negative length-three "
            "vector summing to one"
        )
    if (
        isinstance(palette_prototype_index, bool)
        or not isinstance(palette_prototype_index, int)
        or palette_prototype_index < 0
    ):
        raise ValueError("palette_prototype_index must be a non-negative integer")
    if not isinstance(material_path, str) or not material_path.startswith("/"):
        raise ValueError("material_path must be an absolute USD prim path")
    for name, value in (
        ("pour_event_id", pour_event_id),
        ("birth_event_id", birth_event_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be non-empty")
    if isinstance(birth_step, bool) or not isinstance(birth_step, int) or birth_step <= 0:
        raise ValueError("birth_step must be a positive integer")
    return SourceState(
        scheduled_tank=scheduled_tank,
        composition=tuple(float(value) for value in values),
        palette_prototype_index=palette_prototype_index,
        material_path=material_path,
        pour_event_id=pour_event_id,
        birth_event_id=birth_event_id,
        birth_step=birth_step,
    )


def build_runtime_particle_configuration(
    *,
    solver_position_iterations: int = 4,
    max_particles: int | None = None,
    resolution_profile_id: str | None = None,
) -> dict[str, object]:
    """Return the Isaac runtime settings sourced from Task 1's single owner."""

    if (
        isinstance(solver_position_iterations, bool)
        or not isinstance(solver_position_iterations, int)
        or solver_position_iterations <= 0
    ):
        raise ValueError("solver_position_iterations must be a positive integer")
    if max_particles is None:
        resolved_max_particles = required_particle_capacity()
    elif (
        isinstance(max_particles, bool)
        or not isinstance(max_particles, int)
        or max_particles <= 0
    ):
        raise ValueError("max_particles must be a positive integer or None")
    else:
        resolved_max_particles = max_particles

    profile = particle_resolution_profile(
        resolution_profile_id or "c20_2p5mm"
    )
    return {
        "spawn_spacing_m": profile["spawn_spacing_m"],
        "fluid_rest_offset_m": profile["fluid_rest_offset_m"],
        "particle_contact_offset_m": profile["particle_contact_offset_m"],
        "contact_offset_m": profile["contact_offset_m"],
        "rest_offset_m": profile["rest_offset_m"],
        "solid_rest_offset_m": profile["solid_rest_offset_m"],
        "density_kg_m3": DENSITY_KG_M3,
        "particle_set_mass_override": PARTICLE_SET_MASS_OVERRIDE,
        "max_particles": resolved_max_particles,
        "physics_hz": 120,
        "solver_position_iterations": solver_position_iterations,
        "max_neighborhood": 96,
        "max_velocity_m_s": MAX_VELOCITY_M_S,
        "ccd": True,
        "smoothing": True,
        "anisotropy": True,
        "fabric_enabled": False,
        "positions_writeback_required": True,
        "velocities_writeback_required": True,
    }


def build_resolution_aware_source_sample(
    *,
    particle_count: object,
    reference_particle_count: object,
    resolution_profile_id: str,
    source_velocity_m_s: Sequence[object],
) -> dict[str, object]:
    """Fit a finer event sample to its frozen C20 source boundary.

    NVIDIA's installed ``FluidBallEmitterDemo.particle_sphere`` derives all
    lattice dimensions from fixed physical radius and particle spacing.  This
    cylindrical-nozzle analogue derives three lattice dimensions from the
    unjittered C20 physical support, then greedily selects the farthest lattice
    row so the exact scheduled count avoids concentrated overlaps.  Support
    and centroid use explicit one-spacing discretization bounds.  The old
    sampler's +/-5% spacing jitter is deliberately omitted and recorded as a
    source-sampler correction rather than hidden as a one-variable ablation.
    """

    for name, value in (
        ("particle_count", particle_count),
        ("reference_particle_count", reference_particle_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if resolution_profile_id != C21_RESOLUTION_PROFILE_ID:
        raise ValueError("resolution-aware source sampling is frozen to C21")
    velocity = np.asarray(source_velocity_m_s)
    if velocity.dtype.kind not in "fiu":
        raise TypeError("source_velocity_m_s must contain real numbers")
    velocity = np.asarray(source_velocity_m_s, dtype=np.float64)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
        raise ValueError("source_velocity_m_s must be one finite length-three vector")

    profile = particle_resolution_profile(resolution_profile_id)
    reference_spacing = SPAWN_SPACING_M
    reference = np.asarray(
        [
            (
                ((index % 3) - 1) * reference_spacing,
                (((index // 3) % 3) - 1) * reference_spacing,
                -(index // 9) * reference_spacing,
            )
            for index in range(reference_particle_count)
        ],
        dtype=np.float64,
    )
    lower = np.min(reference, axis=0)
    upper = np.max(reference, axis=0)
    centroid = np.mean(reference, axis=0)
    support = upper - lower
    spacing = float(profile["spawn_spacing_m"])
    reference_physical_support = support + 2.0 * FLUID_REST_OFFSET_M
    target_center_support = np.maximum(
        reference_physical_support - 2.0 * float(profile["render_radius_m"]),
        np.asarray([0.0, 0.0, spacing], dtype=np.float64),
    )
    dimensions = [
        1 if span == 0.0 else int(math.ceil(float(span) / spacing)) + 1
        for span in target_center_support
    ]
    axes = [
        np.linspace(
            float(centroid[axis] - target_center_support[axis] / 2.0),
            float(centroid[axis] + target_center_support[axis] / 2.0),
            dimensions[axis],
        )
        for axis in range(3)
    ]
    candidates = np.asarray(
        np.meshgrid(*axes, indexing="ij"), dtype=np.float64
    ).reshape(3, -1).T
    if particle_count > len(candidates):
        raise RuntimeError(
            "C21 source count exceeds its resolution-derived lattice capacity"
        )
    if particle_count == 1:
        selected = [
            int(np.argmin(np.linalg.norm(candidates - centroid[None, :], axis=1)))
        ]
    else:
        selected = [0, len(candidates) - 1]
        selected_set = set(selected)
        while len(selected) < particle_count:
            remaining = [
                index for index in range(len(candidates)) if index not in selected_set
            ]
            distances = np.linalg.norm(
                candidates[remaining, None, :]
                - candidates[selected][None, :, :],
                axis=2,
            ).min(axis=1)
            chosen = remaining[int(np.argmax(distances))]
            selected.append(chosen)
            selected_set.add(chosen)
    positions = candidates[selected]
    observed_support = np.ptp(positions, axis=0)
    observed_physical_support = (
        observed_support + 2.0 * float(profile["render_radius_m"])
    )
    observed_centroid = np.mean(positions, axis=0)
    centroid_delta = observed_centroid - centroid
    support_delta = observed_physical_support - reference_physical_support
    if particle_count > 1:
        pairwise = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :], axis=2
        )
        pairwise[pairwise == 0.0] = np.inf
        minimum_center_separation_m = float(np.min(pairwise))
    else:
        minimum_center_separation_m = math.inf
    current_equivalent_volume = (
        particle_count * float(profile["effective_particle_volume_ml"])
    )
    reference_equivalent_volume = (
        reference_particle_count * MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML
    )
    volume_delta = current_equivalent_volume - reference_equivalent_volume
    support_gate = bool(np.all(np.abs(support_delta) <= spacing + 1.0e-15))
    centroid_gate = bool(np.all(np.abs(centroid_delta) <= spacing / 2.0 + 1.0e-15))
    separation_gate = bool(
        minimum_center_separation_m >= 0.75 * spacing - 1.0e-15
    )
    flux_tolerance_ml = max(
        MEASURED_PACKING_EFFECTIVE_PARTICLE_VOLUME_ML,
        float(profile["effective_particle_volume_ml"]),
    )
    flux_gate = bool(abs(volume_delta) <= flux_tolerance_ml)
    if not support_gate or not centroid_gate or not separation_gate or not flux_gate:
        raise RuntimeError("C21 source-boundary attestation failed")
    return {
        "local_positions_m": positions.tolist(),
        "attestation": {
            "resolution_profile_id": resolution_profile_id,
            "reference_geometry": "c20_unjittered_3x3_lattice",
            "reference_particle_count": reference_particle_count,
            "particle_count": particle_count,
            "lattice_dimensions": dimensions,
            "reference_center_support_m": support.tolist(),
            "reference_physical_support_m": reference_physical_support.tolist(),
            "target_center_support_m": target_center_support.tolist(),
            "observed_center_support_m": observed_support.tolist(),
            "observed_physical_support_m": observed_physical_support.tolist(),
            "support_delta_m": support_delta.tolist(),
            "reference_centroid_m": centroid.tolist(),
            "observed_centroid_m": observed_centroid.tolist(),
            "centroid_delta_m": centroid_delta.tolist(),
            "source_velocity_m_s": velocity.tolist(),
            "reference_event_equivalent_volume_ml": reference_equivalent_volume,
            "event_equivalent_volume_ml": current_equivalent_volume,
            "event_equivalent_volume_delta_ml": volume_delta,
            "flux_discretization_tolerance_ml": flux_tolerance_ml,
            "support_discretization_tolerance_m": spacing,
            "centroid_discretization_tolerance_m": spacing / 2.0,
            "minimum_center_separation_m": minimum_center_separation_m,
            "minimum_center_separation_ratio": 0.75,
            "legacy_jitter_removed_max_m": 0.05 * reference_spacing,
            "support_gate_passed": support_gate,
            "centroid_gate_passed": centroid_gate,
            "separation_gate_passed": separation_gate,
            "flux_gate_passed": flux_gate,
            "authorized_deviation": (
                "resolution-aware source sampler replaces C20's count-dependent "
                "3x3 axial stack and stochastic jitter"
            ),
            "documented_pattern": (
                "NVIDIA FluidBallEmitterDemo.particle_sphere recomputes lattice "
                "dimensions from fixed physical radius and particle spacing"
            ),
        },
    }


def build_count_driven_liquid_body_geometry(
    source_points: object,
    *,
    contained_particle_count: int,
    particle_volume_m3: float = COUNT_DRIVEN_PARTICLE_VOLUME_M3,
    source_world_height_m: float,
    source_effective_area_m2: float,
) -> dict[str, object]:
    """Build the count-driven twin-liquid body geometry contract."""

    rows = np.asarray(source_points, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1:] != (3,) or not np.all(np.isfinite(rows)):
        raise ValueError("source_points must be finite with shape (N, 3)")
    if len(rows) < 4:
        raise ValueError("source_points must contain a closed-mesh point set")
    if (
        isinstance(contained_particle_count, bool)
        or not isinstance(contained_particle_count, int)
        or contained_particle_count < 0
    ):
        raise ValueError("contained_particle_count must be a non-negative integer")
    source_height = float(source_world_height_m)
    source_area = float(source_effective_area_m2)
    particle_volume = float(particle_volume_m3)
    if (
        not math.isfinite(source_height)
        or source_height <= 0.0
        or not math.isfinite(source_area)
        or source_area <= 0.0
        or not math.isfinite(particle_volume)
        or particle_volume <= 0.0
    ):
        raise ValueError(
            "source world height, effective area and particle volume must be positive"
        )

    target_volume = contained_particle_count * particle_volume
    target_height = target_volume / COUNT_DRIVEN_DODECAGON_AREA_M2
    z_scale = target_height / source_height
    xy_scale = math.sqrt(COUNT_DRIVEN_DODECAGON_AREA_M2 / source_area)
    center_xy = (np.min(rows[:, :2], axis=0) + np.max(rows[:, :2], axis=0)) / 2.0
    bottom_z = float(np.min(rows[:, 2]))
    transformed = rows.copy()
    transformed[:, :2] = center_xy + (rows[:, :2] - center_xy) * xy_scale
    transformed[:, 2] = bottom_z + (rows[:, 2] - bottom_z) * z_scale
    expected_volume = source_area * xy_scale**2 * target_height
    digest = hashlib.sha256(
        np.asarray(transformed, dtype="<f8", order="C").tobytes()
    ).hexdigest()
    return {
        "representation": "count_driven_twin_liquid_mesh",
        "contained_particle_count": contained_particle_count,
        "particle_volume_m3": particle_volume,
        "cross_section_area_m2": COUNT_DRIVEN_DODECAGON_AREA_M2,
        "source_effective_area_m2": source_area,
        "source_world_height_m": source_height,
        "target_volume_m3": target_volume,
        "target_height_m": target_height,
        "xy_area_correction_scale": xy_scale,
        "z_scale": z_scale,
        "visible": contained_particle_count > 0,
        "points": transformed.tolist(),
        "points_sha256": digest,
        "expected_post_transform_volume_m3": expected_volume,
        "expected_volume_residual_m3": expected_volume - target_volume,
    }


def signed_mesh_volume_m3(
    points: object,
    face_vertex_counts: object,
    face_vertex_indices: object,
) -> float:
    """Return signed volume for one closed polygon mesh in metre coordinates."""

    vertices = np.asarray(points, dtype=np.float64)
    counts = np.asarray(face_vertex_counts)
    indices = np.asarray(face_vertex_indices)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("points must be finite with shape (N, 3)")
    if counts.ndim != 1 or counts.dtype.kind not in "iu" or np.any(counts < 3):
        raise ValueError("face_vertex_counts must contain integers of at least three")
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError("face_vertex_indices must contain integers")
    if int(np.sum(counts)) != len(indices):
        raise ValueError("face topology corner count does not match indices")
    if len(indices) and (int(np.min(indices)) < 0 or int(np.max(indices)) >= len(vertices)):
        raise ValueError("face topology contains an out-of-range point index")

    volume = 0.0
    offset = 0
    for count_value in counts.tolist():
        face = indices[offset : offset + int(count_value)]
        offset += int(count_value)
        anchor = vertices[int(face[0])]
        for corner in range(1, len(face) - 1):
            second = vertices[int(face[corner])]
            third = vertices[int(face[corner + 1])]
            volume += float(np.dot(anchor, np.cross(second, third))) / 6.0
    return volume


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _artifact_manifest_valid(
    paths: object,
    hashes: object,
    *,
    count: int,
) -> bool:
    return bool(
        isinstance(paths, list)
        and isinstance(hashes, list)
        and len(paths) == len(hashes) == count
        and all(isinstance(path, str) and path for path in paths)
        and len(set(paths)) == count
        and all(_valid_sha256(value) for value in hashes)
    )


RENDER_STATE_DUPLICATE_POLICY = (
    "byte_identical_rgb_allowed_only_for_post_fill_stream_hidden_equivalence"
)
RENDER_STATE_DUPLICATE_LIMITATION = (
    "the count-driven fallback renders aggregate body geometry and colour rather "
    "than spatial particle composition; exact repeats do not prove physical stasis"
)
_RENDER_STATE_BODY_HASH_FIELDS = (
    "body_points_sha256",
    "body_extent_sha256",
    "body_material_color_sha256",
    "stream_visibility_sha256",
)


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def copy_semantic_effective_render_state(
    record: Mapping[str, object], *, count_driven: bool
) -> dict[str, object]:
    """Copy authored render state into disk-reloaded semantic evidence."""

    glass = record.get("rgb_glass_annotation_state")
    if (
        not isinstance(glass, Mapping)
        or not isinstance(glass.get("prim_path"), str)
        or not glass.get("prim_path")
        or not isinstance(glass.get("attribute_name"), str)
        or not glass.get("attribute_name")
        or glass.get("requested_value") is not False
        or glass.get("composed_type_name") != "bool"
        or glass.get("composed_value") is not False
        or glass.get("property_stack_strongest_is_session_layer") is not True
        or glass.get("source_equivalent") is not True
    ):
        raise ValueError("RGB glass annotation state is missing or invalid")
    result: dict[str, object] = {
        "rgb_glass_annotation_state": json.loads(
            json.dumps(dict(glass), sort_keys=True, allow_nan=False)
        )
    }
    if not count_driven:
        return result

    body = record.get("body_material_render_state")
    body_sha256 = record.get("body_material_render_state_sha256")
    if (
        not isinstance(body, Mapping)
        or not _valid_sha256(body_sha256)
        or body_sha256 != _canonical_json_sha256(dict(body))
    ):
        raise ValueError("body material render state is missing or hash-mismatched")
    stream = record.get("stream_geometry_render_state")
    stream_sha256 = record.get("stream_geometry_render_state_sha256")
    if (
        not isinstance(stream, Mapping)
        or not _valid_sha256(stream_sha256)
        or stream_sha256 != _canonical_json_sha256(dict(stream))
    ):
        raise ValueError("stream geometry render state is missing or hash-mismatched")
    result.update(
        {
            "body_material_render_state": json.loads(
                json.dumps(dict(body), sort_keys=True, allow_nan=False)
            ),
            "body_material_render_state_sha256": body_sha256,
            "stream_geometry_render_state": json.loads(
                json.dumps(dict(stream), sort_keys=True, allow_nan=False)
            ),
            "stream_geometry_render_state_sha256": stream_sha256,
        }
    )
    return result


def _finite_vector(value: object, *, length: int) -> bool:
    return bool(
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_finite_number(component) for component in value)
    )


def _validated_camera_effective_render_state(
    camera: Mapping[str, object],
) -> dict[str, object]:
    state = camera.get("effective_render_state")
    if not isinstance(state, Mapping):
        raise ValueError("camera effective render state is missing")
    matrix = state.get("world_transform_matrix")
    if (
        not isinstance(matrix, (list, tuple))
        or len(matrix) != 4
        or any(not _finite_vector(row, length=4) for row in matrix)
    ):
        raise ValueError("camera effective render state has an invalid world transform")
    projection = state.get("projection")
    if not isinstance(projection, str) or not projection:
        raise ValueError("camera effective render state has no projection")
    positive_fields = (
        "focal_length",
        "horizontal_aperture",
        "vertical_aperture",
    )
    if any(
        not _finite_number(state.get(field)) or float(state[field]) <= 0.0
        for field in positive_fields
    ):
        raise ValueError("camera effective render state has invalid projection attributes")
    for field in ("focus_distance", "f_stop", "time_code"):
        if not _finite_number(state.get(field)) or float(state[field]) < 0.0:
            raise ValueError("camera effective render state has invalid projection attributes")
    clipping = state.get("clipping_range")
    if (
        not _finite_vector(clipping, length=2)
        or float(clipping[0]) <= 0.0
        or float(clipping[1]) <= float(clipping[0])
        or list(clipping) != camera.get("clipping_range_effective")
    ):
        raise ValueError("camera effective render state has invalid clipping range")
    return json.loads(json.dumps(state, sort_keys=True, allow_nan=False))


def _validated_effective_light_state(scene: Mapping[str, object]) -> dict[str, object]:
    state = scene.get("effective_light_state")
    if not isinstance(state, Mapping) or set(state) != set(EXPECTED_LIGHT_PATHS):
        raise ValueError("effective light state must contain every reviewed twin light")
    result: dict[str, object] = {}
    for path in EXPECTED_LIGHT_PATHS:
        row = state.get(path)
        spec = LIGHT_SPECS[path]
        if (
            not isinstance(row, Mapping)
            or row.get("type_name") != spec["type_name"]
            or not _finite_number(row.get("intensity"))
            or float(row["intensity"]) <= 0.0
            or not _finite_number(row.get("exposure"))
            or row.get("transform_path") != spec["transform_path"]
        ):
            raise ValueError(f"effective light state is invalid for {path}")
        if spec["transform_path"] is None:
            if row.get("translate") is not None or row.get("rotate_xyz_degrees") is not None:
                raise ValueError(f"effective light state is invalid for {path}")
        elif (
            not _finite_vector(row.get("translate"), length=3)
            or not _finite_vector(row.get("rotate_xyz_degrees"), length=3)
        ):
            raise ValueError(f"effective light state is invalid for {path}")
        result[path] = json.loads(json.dumps(row, sort_keys=True, allow_nan=False))
    return result


def _validated_rgb_glass_annotation_state(
    value: object,
    *,
    frame_index: int,
    expected_prim_path: str,
    expected_attribute_name: str,
) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("prim_path"), str)
        or value.get("prim_path") != expected_prim_path
        or not isinstance(value.get("attribute_name"), str)
        or value.get("attribute_name") != expected_attribute_name
        or value.get("requested_value") is not False
        or value.get("composed_type_name") != "bool"
        or value.get("composed_value") is not False
        or value.get("property_stack_strongest_is_session_layer") is not True
        or value.get("source_equivalent") is not True
    ):
        raise ValueError(
            f"frame {frame_index} has invalid RGB glass annotation state"
        )
    return dict(value)


def _validated_count_driven_appearance_contract(
    receipt: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    material = receipt.get("material")
    body = receipt.get("count_driven_liquid_body")
    if not isinstance(material, Mapping) or not isinstance(body, Mapping):
        raise ValueError("count-driven appearance contract is missing")
    global_path = material.get("global_liquid_path")
    family = material.get("omnisurface_parameter_family")
    tank_colors = material.get("tank_transmission_colors")
    expected_family = {
        "diffuse_reflection_weight": 0.0,
        "enable_opacity": False,
        "specular_reflection_ior": 1.333,
        "specular_reflection_roughness": 0.08,
        "specular_transmission_scattering_depth_m": 0.01,
        "specular_transmission_weight": 1.0,
        "thin_walled": False,
    }
    if (
        not isinstance(global_path, str)
        or not global_path.startswith("/")
        or not isinstance(family, Mapping)
        or dict(family) != expected_family
    ):
        raise ValueError("count-driven OmniSurface appearance family is invalid")
    if (
        not isinstance(tank_colors, Mapping)
        or set(tank_colors) != set(COMPOSITION_COMPONENTS)
        or any(
            not _finite_vector(tank_colors.get(tank), length=3)
            or any(not 0.0 <= float(value) <= 1.0 for value in tank_colors[tank])
            for tank in COMPOSITION_COMPONENTS
        )
        or material.get("transmission_colors_uncalibrated") is not True
        or material.get("vial_representation") != COUNT_DRIVEN_RENDER_PATH
        or material.get("stream_representation")
        != "render_only_mesh_current_tank_material"
        or material.get("global_blend_rule")
        != "mean_composition_beer_law_transmission_uncalibrated"
    ):
        raise ValueError("count-driven tank appearance contract is invalid")
    body_paths = {
        "representation": body.get("representation"),
        "body_xform_path": body.get("body_xform_path"),
        "body_mesh_path": body.get("body_mesh_path"),
        "material_path": body.get("material_path"),
    }
    if (
        body_paths["representation"] != COUNT_DRIVEN_RENDER_PATH
        or any(
            not isinstance(body_paths[field], str)
            or not body_paths[field].startswith("/")
            for field in ("body_xform_path", "body_mesh_path", "material_path")
        )
        or body_paths["material_path"] != global_path
    ):
        raise ValueError("count-driven body material binding contract is invalid")
    material_contract = {
        "global_liquid_path": global_path,
        "omnisurface_parameter_family": dict(family),
        "tank_transmission_colors": {
            tank: list(tank_colors[tank]) for tank in COMPOSITION_COMPONENTS
        },
        "transmission_colors_uncalibrated": True,
        "vial_representation": material["vial_representation"],
        "stream_representation": material["stream_representation"],
        "global_blend_rule": material["global_blend_rule"],
    }
    return material_contract, body_paths


def _validated_body_material_render_state(
    pair: Mapping[str, object],
    *,
    frame_index: int,
    material_contract: Mapping[str, object],
    body_paths: Mapping[str, object],
) -> dict[str, object]:
    state = pair.get("body_material_render_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"frame {frame_index} body material render state is missing")
    inputs = state.get("shader_inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError(f"frame {frame_index} OmniSurface input payload is missing")
    expected_names = {
        "enable_specular_transmission",
        "specular_transmission_weight",
        "specular_transmission_color",
        "specular_transmission_scattering_depth",
        "thin_walled",
        "diffuse_reflection_weight",
        "specular_reflection_ior",
        "specular_reflection_roughness",
        "enable_opacity",
        "geometry_opacity",
        "geometry_opacity_threshold",
    }
    family = material_contract["omnisurface_parameter_family"]
    assert isinstance(family, Mapping)
    expected_fixed = {
        "enable_specular_transmission": True,
        "specular_transmission_weight": family["specular_transmission_weight"],
        "specular_transmission_scattering_depth": family[
            "specular_transmission_scattering_depth_m"
        ],
        "thin_walled": family["thin_walled"],
        "diffuse_reflection_weight": family["diffuse_reflection_weight"],
        "specular_reflection_ior": family["specular_reflection_ior"],
        "specular_reflection_roughness": family[
            "specular_reflection_roughness"
        ],
        "enable_opacity": family["enable_opacity"],
        "geometry_opacity": 1.0,
        "geometry_opacity_threshold": 0.0,
    }
    if set(inputs) != expected_names:
        raise ValueError(f"frame {frame_index} OmniSurface input payload is incomplete")
    for name, expected in expected_fixed.items():
        actual = inputs.get(name)
        if type(expected) is bool:
            matches = actual is expected
        else:
            matches = _finite_number(actual) and math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-6
            )
        if not matches:
            raise ValueError(f"frame {frame_index} OmniSurface input {name} drifted")
    color = inputs.get("specular_transmission_color")
    if (
        not _finite_vector(color, length=3)
        or any(not 0.0 <= float(value) <= 1.0 for value in color)
        or state.get("body_mesh_path") != body_paths["body_mesh_path"]
        or state.get("material_path") != material_contract["global_liquid_path"]
        or state.get("shader_path")
        != f"{material_contract['global_liquid_path']}/OmniSurface"
        or state.get("shader_inputs_sha256") != _canonical_json_sha256(dict(inputs))
        or pair.get("body_material_color_sha256")
        != _canonical_json_sha256(list(color))
        or pair.get("body_material_render_state_sha256")
        != _canonical_json_sha256(dict(state))
    ):
        raise ValueError(f"frame {frame_index} body material binding readback is invalid")
    return json.loads(json.dumps(state, sort_keys=True, allow_nan=False))


def _validated_stream_geometry_render_state(
    pair: Mapping[str, object],
    *,
    frame_index: int,
) -> dict[str, object]:
    state = pair.get("stream_geometry_render_state")
    if not isinstance(state, Mapping):
        raise ValueError(f"frame {frame_index} stream geometry render state is missing")
    visible = pair.get("visible_stream_tanks")
    active_source = state.get("active_source")
    visibility = state.get("visibility")
    raw_state = {key: value for key, value in state.items() if key != "state_sha256"}
    if (
        state.get("prim_path") != PERSISTENT_RENDER_STREAM_PATH
        or not _finite_number(state.get("height_m"))
        or float(state["height_m"]) <= 0.0
        or not _finite_number(state.get("radius_m"))
        or float(state["radius_m"]) <= 0.0
        or state.get("axis") not in {"Z", "z"}
        or not _finite_vector(state.get("translate"), length=3)
        or state.get("state_sha256") != _canonical_json_sha256(raw_state)
        or pair.get("stream_geometry_render_state_sha256")
        != _canonical_json_sha256(dict(state))
    ):
        raise ValueError(f"frame {frame_index} stream geometry render state is invalid")
    if visible:
        expected_source = visible[0]
        material_state = pair.get("stream_material_state")
        if (
            visibility == "invisible"
            or active_source != expected_source
            or not isinstance(material_state, Mapping)
            or state.get("material_path")
            != material_state.get("readback_material_path")
        ):
            raise ValueError(f"frame {frame_index} visible stream geometry state drifted")
    elif visibility != "invisible" or active_source is not None:
        raise ValueError(f"frame {frame_index} hidden stream geometry state drifted")
    return json.loads(json.dumps(state, sort_keys=True, allow_nan=False))


def build_render_state_duplicate_evidence(
    receipt: Mapping[str, object],
) -> dict[str, object]:
    """Classify exact RGB repeats against effective authored render inputs.

    Count-driven rendering can legitimately produce the same bytes after the
    body geometry, global body colour and hidden stream stop changing.  A
    repeat is allowed only after fill end, with every stream hidden, equal
    effective render fingerprints and equal liquid masks.  Active-pour repeats
    always fail.  Particle compositions are deliberately excluded because this
    fallback renders their aggregate body colour, not spatial particle colour.
    """

    render = receipt.get("render")
    semantics = receipt.get("semantics")
    timing = receipt.get("timing")
    camera = receipt.get("camera")
    scene = receipt.get("scene_contract")
    render = render if isinstance(render, Mapping) else {}
    semantics = semantics if isinstance(semantics, Mapping) else {}
    timing = timing if isinstance(timing, Mapping) else {}
    frame_hashes = render.get("frame_sha256")
    semantic_pairs = semantics.get("pairs")
    fill_end_step = timing.get("fill_end_step")
    if (
        not isinstance(frame_hashes, list)
        or not frame_hashes
        or not isinstance(semantic_pairs, list)
        or len(semantic_pairs) != len(frame_hashes)
        or isinstance(fill_end_step, bool)
        or not isinstance(fill_end_step, int)
        or fill_end_step <= 0
        or not isinstance(camera, Mapping)
        or not isinstance(scene, Mapping)
    ):
        raise ValueError("render-state evidence is missing its frozen run contract")

    source_layers = scene.get("source_layer_records")
    if (
        not isinstance(source_layers, list)
        or not source_layers
        or any(
            not isinstance(row, Mapping) or not _valid_sha256(row.get("sha256"))
            for row in source_layers
        )
    ):
        raise ValueError("render-state evidence has no hashed source scene layers")
    camera_state = _validated_camera_effective_render_state(camera)
    light_state = _validated_effective_light_state(scene)
    annotation_override = semantics.get("transparent_annotation_override")
    if (
        not isinstance(annotation_override, Mapping)
        or not isinstance(annotation_override.get("prim_path"), str)
        or not annotation_override.get("prim_path")
        or not isinstance(annotation_override.get("attribute_name"), str)
        or not annotation_override.get("attribute_name")
        or annotation_override.get("rgb_and_restore_value") is not False
    ):
        raise ValueError("RGB glass annotation contract is missing")
    material_contract, body_paths = _validated_count_driven_appearance_contract(
        receipt
    )
    static_inputs = {
        "render_path": receipt.get("render_path"),
        "source_layer_sha256": [row["sha256"] for row in source_layers],
        "scene": {
            "default_prim": scene.get("default_prim"),
            "meters_per_unit": scene.get("meters_per_unit"),
            "up_axis": scene.get("up_axis"),
            "frozen_time_code": scene.get("frozen_time_code"),
            "hidden_source_visibility": scene.get("hidden_source_visibility"),
            "custom_camera_count": scene.get("custom_camera_count"),
            "custom_light_count": scene.get("custom_light_count"),
            "custom_backdrop_count": scene.get("custom_backdrop_count"),
        },
        "camera_effective_render_state": camera_state,
        "effective_light_state": light_state,
        "count_driven_material_contract": material_contract,
        "count_driven_body_paths": body_paths,
        "sampling": {
            "renderer": render.get("renderer"),
            "resolution": render.get("resolution"),
            "rt_subframes": render.get("rt_subframes"),
            "pathtracing_total_spp_readback": render.get(
                "pathtracing_total_spp_readback"
            ),
            "warmup_updates": render.get("warmup_updates"),
        },
        "nuisance_profile_binding": receipt.get("nuisance_profile_binding"),
        "nuisance_runtime": receipt.get("nuisance_runtime"),
    }
    static_sha256 = _canonical_json_sha256(static_inputs)

    fingerprints: list[str] = []
    physics_steps: list[int] = []
    mask_hashes: list[str] = []
    stream_hidden: list[bool] = []
    for index, (rgb_sha256, pair) in enumerate(zip(frame_hashes, semantic_pairs)):
        if not _valid_sha256(rgb_sha256):
            raise ValueError(f"frame {index} has an invalid RGB hash")
        if not isinstance(pair, Mapping) or pair.get("rgb_sha256") != rgb_sha256:
            raise ValueError(f"frame {index} RGB hash is not bound to its semantic pair")
        step = pair.get("physics_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"frame {index} has an invalid physics step")
        if any(not _valid_sha256(pair.get(field)) for field in _RENDER_STATE_BODY_HASH_FIELDS):
            raise ValueError(f"frame {index} is missing an authored render-state hash")
        body_visibility = pair.get("body_visibility")
        representation = pair.get("liquid_representation")
        visible_streams = pair.get("visible_stream_tanks")
        if not isinstance(body_visibility, str) or not body_visibility:
            raise ValueError(f"frame {index} has invalid body visibility")
        if not isinstance(representation, str) or not representation:
            raise ValueError(f"frame {index} has invalid liquid representation")
        mask_sha256 = pair.get("mask_sha256")
        if not _valid_sha256(mask_sha256):
            raise ValueError(f"frame {index} has an invalid liquid mask hash")
        if (
            not isinstance(visible_streams, list)
            or len(visible_streams) > 1
            or any(not isinstance(value, str) or not value for value in visible_streams)
        ):
            raise ValueError(f"frame {index} has invalid visible stream state")
        stream_material = pair.get("stream_material_state")
        if not isinstance(stream_material, Mapping):
            raise ValueError(f"frame {index} has no stream material state")
        visible_material: dict[str, object] | None = None
        if visible_streams:
            source = visible_streams[0]
            requested = stream_material.get("requested_material_path")
            readback = stream_material.get("readback_material_path")
            if (
                stream_material.get("source_tank") != source
                or not isinstance(requested, str)
                or not requested
                or readback != requested
                or stream_material.get("matched") is not True
            ):
                raise ValueError(f"frame {index} has inconsistent visible stream material")
            visible_material = {
                "source_tank": source,
                "requested_material_path": requested,
                "readback_material_path": readback,
            }
        glass_state = _validated_rgb_glass_annotation_state(
            pair.get("rgb_glass_annotation_state"),
            frame_index=index,
            expected_prim_path=annotation_override["prim_path"],
            expected_attribute_name=annotation_override["attribute_name"],
        )
        body_material_state = _validated_body_material_render_state(
            pair,
            frame_index=index,
            material_contract=material_contract,
            body_paths=body_paths,
        )
        stream_geometry_state = _validated_stream_geometry_render_state(
            pair,
            frame_index=index,
        )
        state = {
            "static_effective_render_inputs_sha256": static_sha256,
            "liquid_representation": representation,
            "body_points_sha256": pair["body_points_sha256"],
            "body_extent_sha256": pair["body_extent_sha256"],
            "body_visibility": body_visibility,
            "body_material_color_sha256": pair["body_material_color_sha256"],
            "stream_visibility_sha256": pair["stream_visibility_sha256"],
            "visible_stream_tanks": visible_streams,
            "visible_stream_material": visible_material,
            "rgb_glass_annotation_state": glass_state,
            "body_material_render_state_sha256": pair[
                "body_material_render_state_sha256"
            ],
            "stream_geometry_render_state_sha256": pair[
                "stream_geometry_render_state_sha256"
            ],
        }
        fingerprints.append(_canonical_json_sha256(state))
        physics_steps.append(step)
        mask_hashes.append(mask_sha256)
        stream_hidden.append(not visible_streams)

    groups: dict[str, list[int]] = {}
    for index, rgb_sha256 in enumerate(frame_hashes):
        groups.setdefault(rgb_sha256, []).append(index)
    repeats: list[dict[str, object]] = []
    for rgb_sha256, indices in groups.items():
        if len(indices) < 2:
            continue
        states = [fingerprints[index] for index in indices]
        distinct_states = list(dict.fromkeys(states))
        masks = [mask_hashes[index] for index in indices]
        all_post_fill = all(physics_steps[index] >= fill_end_step for index in indices)
        all_stream_hidden = all(stream_hidden[index] for index in indices)
        masks_equal = len(set(masks)) == 1
        allowed = (
            len(distinct_states) == 1
            and all_post_fill
            and all_stream_hidden
            and masks_equal
        )
        repeats.append(
            {
                "rgb_sha256": rgb_sha256,
                "frame_indices_zero_based": indices,
                "physics_steps": [physics_steps[index] for index in indices],
                "render_state_sha256": states,
                "distinct_render_state_count": len(distinct_states),
                "mask_sha256": masks,
                "equal_liquid_masks": masks_equal,
                "all_at_or_after_fill_end": all_post_fill,
                "all_stream_hidden": all_stream_hidden,
                "allowed": allowed,
            }
        )
    conflicting = [group for group in repeats if group["allowed"] is not True]
    return {
        "policy": RENDER_STATE_DUPLICATE_POLICY,
        "fingerprint_algorithm": "sha256_canonical_json_v1",
        "fingerprint_schema": {
            "run_level": [
                "render_path",
                "source_layer_sha256",
                "scene",
                "camera_effective_render_state",
                "effective_light_state",
                "count_driven_material_contract",
                "count_driven_body_paths",
                "sampling",
                "nuisance_profile_binding",
                "nuisance_runtime",
            ],
            "per_frame": [
                "liquid_representation",
                "body_points_sha256",
                "body_extent_sha256",
                "body_visibility",
                "body_material_color_sha256",
                "stream_visibility_sha256",
                "visible_stream_tanks",
                "visible_stream_material",
                "rgb_glass_annotation_state",
                "body_material_render_state_sha256",
                "stream_geometry_render_state_sha256",
            ],
            "repeat_gates": [
                "physics_step_at_or_after_fill_end",
                "stream_hidden",
                "equal_liquid_mask_sha256",
            ],
        },
        "fill_end_step": fill_end_step,
        "static_effective_render_inputs_sha256": static_sha256,
        "frame_render_state_sha256": fingerprints,
        "exact_repeat_rgb_groups": repeats,
        "exact_repeat_rgb_group_count": len(repeats),
        "exact_repeat_rgb_frame_count": sum(
            len(group["frame_indices_zero_based"]) for group in repeats
        ),
        "conflicting_rgb_group_count": len(conflicting),
        "passed": not conflicting,
        "limitation": RENDER_STATE_DUPLICATE_LIMITATION,
    }


def _active_stream_for_step(step: int) -> list[str]:
    for _ordinal, tank_id, start_step, end_step in FULL_RUN_STREAM_ROWS:
        if start_step <= step <= end_step:
            return [tank_id]
    return []


def _count_driven_expected_volume_rows(
    receipt: Mapping[str, object],
) -> tuple[tuple[int, float, int], ...]:
    """Derive stage rows from the bound runtime plan, with legacy fallback."""

    if "plan" not in receipt:
        return FULL_RUN_VOLUME_ROWS
    plan = receipt.get("plan")
    pours = plan.get("pours") if isinstance(plan, Mapping) else None
    if not isinstance(pours, list) or not pours:
        return ()
    rows: list[tuple[int, float, int]] = []
    cumulative_volume_ml = 0.0
    previous_count = 0
    for pour in pours:
        if not isinstance(pour, Mapping):
            return ()
        step = pour.get("stage_capture_step")
        volume_ml = pour.get("programmed_volume_ml")
        count = pour.get("cumulative_particle_count")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or not _finite_number(volume_ml)
            or float(volume_ml) <= 0.0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= previous_count
        ):
            return ()
        cumulative_volume_ml += float(volume_ml)
        rows.append((step, cumulative_volume_ml, count))
        previous_count = count
    return tuple(rows)


def _validate_count_driven_body_contract(
    receipt: Mapping[str, object],
) -> tuple[str, ...]:
    """Validate the explicit hybrid PBD-state/count-driven-render contract."""

    if receipt.get("render_path") != COUNT_DRIVEN_RENDER_PATH:
        return ()
    failures: list[str] = []
    body = receipt.get("count_driven_liquid_body")
    body = body if isinstance(body, Mapping) else {}
    expected_paths = {
        "source_xform_path": "/FillStation/FillStation_LocalRoot/Fill_liquid",
        "source_mesh_path": (
            "/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid"
        ),
        "body_xform_path": (
            "/FillStation/FillStation_LocalRoot/FluidmixF2CountDrivenLiquid"
        ),
        "body_mesh_path": (
            "/FillStation/FillStation_LocalRoot/FluidmixF2CountDrivenLiquid/Fill_liquid"
        ),
        "material_path": "/FillStation/FluidmixF2/GlobalLiquidMaterial",
    }
    plan = receipt.get("plan")
    render_treatment = (
        plan.get("render_treatment") if isinstance(plan, Mapping) else None
    )
    expected_particle_volume_m3 = (
        render_treatment.get("particle_volume_m3")
        if isinstance(render_treatment, Mapping)
        else COUNT_DRIVEN_PARTICLE_VOLUME_M3
    )
    numeric = {
        "source_world_bottom_z_m": 0.9762065645,
        "cross_section_area_m2": COUNT_DRIVEN_DODECAGON_AREA_M2,
    }
    source_counts_valid = bool(
        body.get("source_point_count") == 512
        and body.get("source_face_count") == 448
        and body.get("source_corner_count") == 1664
        and body.get("source_fan_triangle_count") == 768
    )
    source_samples_valid = bool(
        body.get("requested_source_time_code") == 686
        and body.get("resolved_points_time_code") == 681
        and body.get("resolved_normals_time_code") == 681
        and body.get("resolved_extent_time_code") == 668
    )
    source_geometry_valid = bool(
        body.get("source_geometry_sha256") == COUNT_DRIVEN_SOURCE_GEOMETRY_SHA256
        and body.get("source_topology_sha256") == COUNT_DRIVEN_SOURCE_TOPOLOGY_SHA256
        and body.get("copied_primvar_names") == []
        and _finite_number(body.get("source_world_height_m"))
        and math.isclose(
            float(body.get("source_world_height_m", math.nan)),
            0.122894103,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        and _finite_number(body.get("source_signed_volume_m3"))
        and math.isclose(
            abs(float(body.get("source_signed_volume_m3", math.nan))),
            66.762882e-6,
            rel_tol=0.0,
            abs_tol=2.0e-10,
        )
        and _finite_number(body.get("source_effective_area_m2"))
        and math.isclose(
            float(body.get("source_effective_area_m2", math.nan)),
            543.255377e-6,
            rel_tol=0.0,
            abs_tol=2.0e-12,
        )
        and _finite_number(body.get("xy_area_correction_scale"))
        and math.isclose(
            float(body.get("xy_area_correction_scale", math.nan)),
            math.sqrt(
                COUNT_DRIVEN_DODECAGON_AREA_M2
                / float(body.get("source_effective_area_m2", math.nan))
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )
    flags_valid = all(
        body.get(key) is expected
        for key, expected in {
            "body_starts_hidden_at_zero_count": True,
            "sphere_instances_hidden": True,
            "physics_point_instancer_preserved": True,
            "pbd_state_independent": True,
            "global_uncalibrated_blend": True,
            "spatial_mixing_color_rendered": False,
            "spatial_layering_color_rendered": False,
            "extent_recomputed_each_update": True,
            "training_eligible": False,
        }.items()
    )
    if (
        body.get("representation") != COUNT_DRIVEN_RENDER_PATH
        or any(body.get(key) != value for key, value in expected_paths.items())
        or any(
            not _finite_number(body.get(key))
            or not math.isclose(
                float(body.get(key, math.nan)),
                value,
                rel_tol=0.0,
                abs_tol=(1.0e-9 if key == "source_world_bottom_z_m" else 1.0e-12),
            )
            for key, value in numeric.items()
        )
        or not _finite_number(expected_particle_volume_m3)
        or float(expected_particle_volume_m3) <= 0.0
        or not _finite_number(body.get("particle_volume_m3"))
        or not math.isclose(
            float(body.get("particle_volume_m3", math.nan)),
            float(expected_particle_volume_m3),
            rel_tol=0.0,
            abs_tol=1.0e-18,
        )
        or body.get("semantic_label") != "liquid"
        or body.get("normals_policy")
        != "omit_authored_normals_after_anisotropic_scale"
        or body.get("geometry_authoring_numeric_path")
        != "local_points_vec3f_then_inherited_world_transform"
        or not _finite_number(
            body.get("post_transform_volume_residual_abs_tolerance_m3")
        )
        or not math.isclose(
            float(
                body.get(
                    "post_transform_volume_residual_abs_tolerance_m3",
                    math.nan,
                )
            ),
            2.0e-10,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not source_counts_valid
        or not source_samples_valid
        or not source_geometry_valid
        or not flags_valid
    ):
        failures.append("count_driven_body_contract")

    qualification = receipt.get("qualification")
    first_stage_only = bool(
        isinstance(qualification, Mapping)
        and qualification.get("first_stage_only") is True
    )
    all_expected_rows = _count_driven_expected_volume_rows(receipt)
    expected_rows = (
        all_expected_rows[:1] if first_stage_only else all_expected_rows
    )
    stages = body.get("stage_geometry")
    stage_rows: list[tuple[int, int]] = []
    stage_geometry_valid = isinstance(stages, list) and len(stages) == len(expected_rows)
    if stage_geometry_valid:
        for stage, expected in zip(stages, expected_rows):
            if not isinstance(stage, Mapping):
                stage_geometry_valid = False
                break
            step, _volume_ml, count = expected
            stage_rows.append((stage.get("physics_step"), stage.get("contained_particle_count")))
            target_volume = count * float(expected_particle_volume_m3)
            target_height = target_volume / COUNT_DRIVEN_DODECAGON_AREA_M2
            world_extent = np.asarray(
                stage.get("world_extent_m"), dtype=np.float64
            )
            if (
                stage.get("physics_step") != step
                or stage.get("contained_particle_count") != count
                or not _finite_number(stage.get("target_volume_m3"))
                or not math.isclose(
                    float(stage.get("target_volume_m3", math.nan)),
                    target_volume,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or not _finite_number(stage.get("target_height_m"))
                or not math.isclose(
                    float(stage.get("target_height_m", math.nan)),
                    target_height,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or not _finite_number(stage.get("post_transform_signed_volume_m3"))
                or not _finite_number(stage.get("post_transform_volume_residual_m3"))
                or stage.get("volume_proof_coordinate_path")
                != "local_points_vec3f_then_inherited_world_transform"
                or abs(float(stage.get("post_transform_volume_residual_m3", math.inf)))
                > 2.0e-10
                or not math.isclose(
                    abs(
                        abs(
                            float(
                                stage.get(
                                    "post_transform_signed_volume_m3", math.nan
                                )
                            )
                        )
                        - target_volume
                    ),
                    abs(
                        float(
                            stage.get(
                                "post_transform_volume_residual_m3", math.nan
                            )
                        )
                    ),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
                or not _valid_sha256(stage.get("points_sha256"))
                or not _valid_sha256(stage.get("extent_sha256"))
                or not _finite_number(stage.get("world_bottom_z_m"))
                or not math.isclose(
                    float(stage.get("world_bottom_z_m", math.nan)),
                    numeric["source_world_bottom_z_m"],
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or world_extent.shape != (2, 3)
                or not np.all(np.isfinite(world_extent))
                or np.any(world_extent[1] <= world_extent[0])
                or not math.isclose(
                    float(world_extent[0, 2]),
                    numeric["source_world_bottom_z_m"],
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not math.isclose(
                    float(world_extent[1, 2] - world_extent[0, 2]),
                    target_height,
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
            ):
                stage_geometry_valid = False
                break
    if not expected_rows or not stage_geometry_valid:
        failures.append("count_driven_body_contract")

    containment = receipt.get("containment")
    capture_records = (
        containment.get("capture_records")
        if isinstance(containment, Mapping)
        else None
    )
    target_containment_valid = isinstance(capture_records, list)
    if target_containment_valid:
        for step, _volume_ml, count in expected_rows:
            matches = [
                row
                for row in capture_records
                if isinstance(row, Mapping) and row.get("physics_step") == step
            ]
            if len(matches) != 1:
                target_containment_valid = False
                break
            row = matches[0]
            if (
                row.get("emitted_particle_count") != count
                or row.get("active_particle_count") != count
                or row.get("inside_vial_interior_count") != count
                or row.get("outside_vial_interior_count") != 0
                or row.get("expected_in_flight_stream_count") != 0
                or row.get("leaked_particle_count") != 0
                or row.get("finite_state") is not True
            ):
                target_containment_valid = False
                break
    if not target_containment_valid:
        failures.append("count_driven_live_containment_contract")

    semantics = receipt.get("semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    control = semantics.get("hidden_particle_control")
    reset = (
        control.get("renderer_accumulation_reset_after_hide")
        if isinstance(control, Mapping)
        else None
    )
    if (
        semantics.get("count_driven_body_semantic_bound") is not True
        or semantics.get("point_instancer_rgb_visible") is not False
        or semantics.get("point_instancer_semantic_visible") is not False
        or not isinstance(control, Mapping)
        or not expected_rows
        or control.get("physics_step") != expected_rows[0][0]
        or control.get("emitted_particle_count") != expected_rows[0][2]
        or control.get("inside_vial_interior_count") != expected_rows[0][2]
        or control.get("body_and_stream_hidden_liquid_pixel_count") != 0
        or not isinstance(reset, Mapping)
        or type(reset.get("supported")) is not bool
        or type(reset.get("called")) is not bool
        or reset.get("called") is not reset.get("supported")
        or not isinstance(control.get("control_semantic_path"), str)
        or not control.get("control_semantic_path")
        or not _valid_sha256(control.get("control_semantic_sha256"))
        or control.get("body_only_target_path") != expected_paths["body_mesh_path"]
        or not isinstance(control.get("body_only_target_ids"), list)
        or not control.get("body_only_target_ids")
        or control.get("body_only_capture_step") != expected_rows[0][0]
        or not isinstance(control.get("body_only_liquid_pixel_count"), int)
        or control.get("body_only_liquid_pixel_count") <= 0
        or control.get("passed") is not True
    ):
        failures.append("count_driven_hidden_particle_control")

    blend = receipt.get("global_blend")
    if (
        not isinstance(blend, Mapping)
        or blend.get("global_blend_rule")
        != "contained_particle_mean_beer_law_transmission_uncalibrated"
        or blend.get("spatial_mixing_color_rendered") is not False
        or blend.get("spatial_layering_color_rendered") is not False
    ):
        failures.append("count_driven_global_blend_contract")
    claims = receipt.get("claims")
    if (
        not isinstance(claims, Mapping)
        or claims.get("particle_composition_rendered") is not False
        or claims.get("spatial_mixing_color_rendered") is not False
        or claims.get("spatial_layering_color_rendered") is not False
        or claims.get("global_vial_blend_rendered") is not True
    ):
        failures.append("count_driven_claim_limit_contract")
    return tuple(dict.fromkeys(failures))


def _visible_height_policy_valid(gate: Mapping[str, object]) -> bool:
    """Accept historical gates or validate the explicit opaque-base policy."""

    policy = gate.get("visible_height_policy")
    if policy is None:
        return True
    programmed_volume = gate.get("programmed_cumulative_volume_ml")
    return bool(
        isinstance(policy, Mapping)
        and policy.get("policy_id") == "real_video5_opaque_base_cutoff_v1"
        and _finite_number(programmed_volume)
        and policy.get("required")
        is (float(programmed_volume) > OPAQUE_BASE_CUTOFF_VOLUME_ML)
        and policy.get("opaque_base_cutoff_volume_ml")
        == OPAQUE_BASE_CUTOFF_VOLUME_ML
        and math.isclose(
            float(policy.get("opaque_base_cutoff_height_m", math.nan)),
            OPAQUE_BASE_CUTOFF_HEIGHT_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and policy.get("comparison_space")
        == "liquid_height_above_opaque_base_cutoff_m"
        and policy.get("source") == OPAQUE_BASE_CUTOFF_SOURCE
        and policy.get("calibrated") is False
    )


def _visible_height_required(gate: Mapping[str, object]) -> bool:
    policy = gate.get("visible_height_policy")
    return (
        bool(policy.get("required"))
        if isinstance(policy, Mapping)
        else True
    )


def _visible_height_contract_valid(
    gate: Mapping[str, object], *, require_evidence: bool
) -> bool:
    if not _visible_height_policy_valid(gate):
        return False
    required = _visible_height_required(gate)
    evidence = gate.get("visible_mask_evidence")
    if required:
        return bool(
            gate.get("visible_passed") is True
            and (
                not require_evidence
                or (
                    isinstance(evidence, Mapping)
                    and evidence.get("required") is True
                    and evidence.get("passed") is True
                )
            )
        )
    return bool(
        gate.get("visible_passed") is None
        and isinstance(evidence, Mapping)
        and evidence.get("required") is False
        and evidence.get("mask_evaluated") is False
        and evidence.get("status")
        == "not_required_at_or_below_opaque_base_cutoff"
    )


def _combined_volume_gate_valid(gate: Mapping[str, object]) -> bool:
    expected_passed = gate.get("state_passed") is True
    return bool(
        gate.get("passed") is expected_passed
        and gate.get("failure_reasons") == []
        and _visible_height_contract_valid(gate, require_evidence=False)
    )


def _stream_instance_semantic_pair_valid(
    pair: Mapping[str, object], expected_streams: object
) -> bool:
    """Validate one persistent stream's instance-to-liquid coverage receipt."""

    expected = (
        [str(value) for value in expected_streams]
        if isinstance(expected_streams, list)
        else []
    )
    if len(expected) > 1:
        return False
    expected_visible = bool(expected)
    overlap = pair.get("stream_semantic_overlap")
    material = pair.get("stream_material_state")
    mapping = pair.get("instance_id_to_prim_path")
    if (
        not isinstance(overlap, Mapping)
        or not isinstance(material, Mapping)
        or not isinstance(mapping, Mapping)
        or any(
            not isinstance(raw_id, str)
            or not raw_id.isdecimal()
            or str(int(raw_id, 10)) != raw_id
            or not isinstance(path, str)
            or not (
                path.startswith("/")
                or (raw_id == "0" and path == "BACKGROUND")
                or (raw_id == "1" and path == "UNLABELLED")
            )
            for raw_id, path in mapping.items()
        )
        or overlap.get("instance_path") != PERSISTENT_RENDER_STREAM_PATH
        or overlap.get("expected_visible") is not expected_visible
        or overlap.get("passed") is not True
    ):
        return False
    instance_ids = overlap.get("instance_ids")
    instance_count = overlap.get("instance_pixel_count")
    overlap_count = overlap.get("liquid_overlap_pixel_count")
    uncovered_count = overlap.get("non_liquid_instance_pixel_count")
    coverage = overlap.get("coverage_fraction")
    if (
        not isinstance(instance_ids, list)
        or any(type(value) is not int or value < 0 for value in instance_ids)
        or len(instance_ids) != len(set(instance_ids))
        or type(instance_count) is not int
        or type(overlap_count) is not int
        or type(uncovered_count) is not int
        or instance_count < 0
        or overlap_count < 0
        or uncovered_count < 0
        or overlap_count + uncovered_count != instance_count
    ):
        return False
    if expected_visible:
        source = expected[0]
        return bool(
            instance_ids
            and instance_count > 0
            and overlap_count == instance_count
            and uncovered_count == 0
            and coverage == 1.0
            and all(
                mapping.get(str(instance_id)) == PERSISTENT_RENDER_STREAM_PATH
                for instance_id in instance_ids
            )
            and material.get("source_tank") == source
            and isinstance(material.get("requested_material_path"), str)
            and bool(material.get("requested_material_path"))
            and material.get("readback_material_path")
            == material.get("requested_material_path")
            and material.get("matched") is True
        )
    readback = material.get("readback_material_path")
    return bool(
        instance_ids == []
        and instance_count == 0
        and overlap_count == 0
        and uncovered_count == 0
        and coverage is None
        and material.get("source_tank") is None
        and material.get("requested_material_path") is None
        and material.get("matched") is None
        and (readback is None or (isinstance(readback, str) and bool(readback)))
    )


def _two_substep_receipt_mode_valid(
    receipt: Mapping[str, object],
    arguments: Mapping[str, object],
    *,
    runtime_final_step: int,
) -> bool:
    """Accept only the isolated qualification or its fully bound train route."""

    particle_system = receipt.get("particle_system")
    common = bool(
        arguments.get("rt_subframes") == 64
        and arguments.get("pathtracing_total_spp") == 64
        and arguments.get("nuisance_receipt") is None
        and arguments.get("nuisance_receipt_sha256") is None
        and arguments.get("nuisance_case_id") is None
        and receipt.get("nuisance_profile_binding") is None
        and receipt.get("nuisance_runtime") is None
        and isinstance(particle_system, Mapping)
        and particle_system.get("runtime_particle_ids_exact_each_report_interval")
        is True
        and particle_system.get("runtime_particle_identity_check_count")
        == runtime_final_step
        and particle_system.get("runtime_particle_id_mismatch_count") == 0
        and _valid_sha256(
            particle_system.get("runtime_particle_ids_final_sha256")
        )
    )
    if not common:
        return False

    train_fields = (
        arguments.get("train_render_plan"),
        arguments.get("train_render_plan_sha256"),
        arguments.get("train_case_id"),
        arguments.get("accepted_qualification_receipt"),
        arguments.get("accepted_qualification_receipt_sha256"),
    )
    any_train_binding = any(value is not None for value in train_fields)
    if not any_train_binding:
        return bool(
            receipt.get("ablation")
            == {
                "id": "c20_two_physx_substeps_per_report_interval",
                "qualification_only": True,
                "training_or_cohort_use_authorized": False,
                "changed_variable": "physics_substeps_per_report_interval",
            }
            and receipt.get("execution_schedule") is None
            and arguments.get("execution_schedule") is None
            and arguments.get("execution_schedule_sha256") is None
            and arguments.get("train_render_plan") is None
            and arguments.get("train_render_plan_sha256") is None
            and arguments.get("train_case_id") is None
            and arguments.get("accepted_qualification_receipt") is None
            and arguments.get("accepted_qualification_receipt_sha256") is None
            and arguments.get("seed") == 20261025
        )

    if not all(isinstance(value, str) and value for value in train_fields):
        return False
    train_plan, train_plan_sha, case_id, qualification_path, qualification_sha = (
        train_fields
    )
    if (
        not _valid_sha256(train_plan_sha)
        or not _valid_sha256(qualification_sha)
        or arguments.get("execution_schedule") is not None
        or arguments.get("execution_schedule_sha256") is not None
        or type(arguments.get("seed")) is not int
    ):
        return False
    execution_schedule = receipt.get("execution_schedule")
    if not isinstance(execution_schedule, Mapping):
        return False
    renderer_binding = execution_schedule.get("renderer_case_binding")
    return bool(
        receipt.get("ablation")
        == {
            "id": "accepted_two_substep_train_case",
            "qualification_only": False,
            "training_or_cohort_use_authorized": True,
            "changed_variable": "physics_substeps_per_report_interval",
            "accepted_qualification_receipt_sha256": qualification_sha,
        }
        and execution_schedule.get("contract_kind") == "train_residual_v2"
        and execution_schedule.get("schedule_path") == train_plan
        and _valid_sha256(execution_schedule.get("schedule_sha256"))
        and _valid_sha256(execution_schedule.get("schedule_file_sha256"))
        and execution_schedule.get("case_id") == case_id
        and isinstance(renderer_binding, Mapping)
        and renderer_binding.get("case_id") == case_id
        and execution_schedule.get("cohort_plan_sha256") == train_plan_sha
        and execution_schedule.get("source_plan_hashes_match") is True
        and execution_schedule.get("model_input_allowlist_passed") is True
        and execution_schedule.get("render_seed") == arguments.get("seed")
        and execution_schedule.get("qualification_receipt_sha256")
        == qualification_sha
        and execution_schedule.get("render_authorized") is True
        and isinstance(qualification_path, str)
        and bool(qualification_path)
    )


def _validate_full_run_contract(receipt: Mapping[str, object]) -> tuple[str, ...]:
    """Validate the frozen A30 full-run denominator and artifact contract."""

    failures: list[str] = []
    count_driven_body = receipt.get("render_path") == COUNT_DRIVEN_RENDER_PATH
    arguments = receipt.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    system = receipt.get("particle_system")
    runtime_contract = bool(
        isinstance(system, Mapping)
        and isinstance(system.get("runtime_configuration"), Mapping)
    )
    runtime_schedule = receipt.get("runtime_capture_schedule")
    runtime_capture_rows = (
        runtime_schedule.get("captures")
        if isinstance(runtime_schedule, Mapping)
        else None
    )
    plan = receipt.get("plan")
    runtime_pours = plan.get("pours") if isinstance(plan, Mapping) else None
    plan_timing = plan.get("timing") if isinstance(plan, Mapping) else None
    bound_runtime_plan = bool(
        runtime_contract
        and isinstance(runtime_capture_rows, list)
        and runtime_capture_rows
        and isinstance(runtime_pours, list)
        and runtime_pours
        and all(isinstance(pour, Mapping) for pour in runtime_pours)
        and isinstance(plan_timing, Mapping)
    )
    if bound_runtime_plan:
        expected_runtime_schedule: dict[str, object] | None = None
        capture_stride = arguments.get("capture_stride")
        try:
            expected_runtime_schedule = build_runtime_capture_schedule(
                plan, capture_stride_steps=capture_stride
            )
        except (TypeError, ValueError, KeyError, OverflowError):
            failures.append("full_run_capture_schedule_contract")
        if expected_runtime_schedule != runtime_schedule:
            failures.append("full_run_capture_schedule_contract")
        effective_runtime_schedule = (
            expected_runtime_schedule
            if expected_runtime_schedule is not None
            else runtime_schedule
        )
        runtime_capture_rows = effective_runtime_schedule["captures"]
        expected_frame_steps = [
            int(row["physics_step"])
            for row in runtime_capture_rows
            if isinstance(row, Mapping) and "physics_step" in row
        ]
        runtime_final_step = int(
            effective_runtime_schedule.get("final_plus_3_0_step", -1)
        )
        stage_steps = [int(pour["stage_capture_step"]) for pour in runtime_pours]
        runtime_volume_rows = []
        expected_streams_by_step: dict[int, list[str]] = {}
        for row in runtime_capture_rows:
            if not isinstance(row, Mapping) or "physics_step" not in row:
                continue
            step = int(row["physics_step"])
            visible_streams = row.get("visible_stream_tanks")
            expected_streams_by_step[step] = (
                [str(value) for value in visible_streams]
                if isinstance(visible_streams, list)
                else []
            )
            if expected_streams_by_step[step] or step not in stage_steps:
                continue
            completed = [
                pour
                for pour in runtime_pours
                if step >= int(pour["end_step"])
            ]
            if completed:
                runtime_volume_rows.append(
                    (
                        step,
                        sum(float(pour["programmed_volume_ml"]) for pour in completed),
                        sum(int(pour["particle_count"]) for pour in completed),
                    )
                )
        segmented_stream_contract = any(
            isinstance(pour.get("emission_segments"), list)
            for pour in runtime_pours
        )
        expected_stream_rows = []
        stream_segment_capture_counts: list[int] = []
        if segmented_stream_contract:
            for pour in runtime_pours:
                segments = pour.get("emission_segments")
                if not isinstance(segments, list):
                    segments = [
                        {
                            "ordinal": 1,
                            "segment_role": "current_tank",
                            "source_tank_id": pour["tank_id"],
                            "particle_count": pour["particle_count"],
                            "start_step": pour["start_step"],
                            "end_step": pour["end_step"],
                        }
                    ]
                for fallback_ordinal, segment in enumerate(segments, start=1):
                    if (
                        not isinstance(segment, Mapping)
                        or int(segment.get("particle_count", 0)) <= 0
                    ):
                        continue
                    segment_start = int(segment["start_step"])
                    segment_end = int(segment["end_step"])
                    source_tank = str(segment["source_tank_id"])
                    segment_ordinal = int(
                        segment.get("ordinal", fallback_ordinal)
                    )
                    expected_stream_rows.append(
                        (
                            int(pour["ordinal"]),
                            segment_ordinal,
                            str(segment.get("segment_role", "current_tank")),
                            str(pour["tank_id"]),
                            source_tank,
                            segment_start,
                            segment_end,
                        )
                    )
                    stream_segment_capture_counts.append(
                        sum(
                            1
                            for row in runtime_capture_rows
                            if isinstance(row, Mapping)
                            and row.get("kind") == "in_pour"
                            and row.get("pour_ordinal") == pour["ordinal"]
                            and segment_start
                            <= int(row["physics_step"])
                            <= segment_end
                            and row.get("visible_stream_tanks")
                            == [source_tank]
                        )
                    )
        else:
            expected_stream_rows = [
                (
                    int(pour["ordinal"]),
                    str(pour["tank_id"]),
                    int(pour["start_step"]),
                    int(pour["end_step"]),
                )
                for pour in runtime_pours
            ]
            stream_segment_capture_counts = []
        expected_physics_hz = int(effective_runtime_schedule["physics_hz"])
        expected_capture_hz = float(effective_runtime_schedule["capture_hz"])
        expected_capture_stride = int(
            effective_runtime_schedule["capture_stride_steps"]
        )
        expected_pour_durations = [
            int(pour["duration_steps"]) for pour in runtime_pours
        ]
        expected_pour_ends = [int(pour["end_step"]) for pour in runtime_pours]
        expected_fill_end = int(plan_timing["fill_end_step"])
        expected_primary_final_step = int(
            plan_timing["primary_final_capture_step"]
        )
        expected_primary_final_offset_s = float(
            plan_timing["primary_final_offset_s"]
        )
        expected_no_inter_pour_settle = bool(
            plan_timing.get("no_inter_pour_settle")
        )
    else:
        expected_frame_steps = list(FULL_RUN_FRAME_STEPS)
        runtime_final_step = 4004
        runtime_volume_rows = FULL_RUN_VOLUME_ROWS
        stage_steps = [row[0] for row in FULL_RUN_VOLUME_ROWS]
        expected_stream_rows = list(FULL_RUN_STREAM_ROWS)
        expected_streams_by_step = {
            step: (
                []
                if index in FULL_RUN_STAGE_INDICES
                else _active_stream_for_step(step)
            )
            for index, step in enumerate(FULL_RUN_FRAME_STEPS)
        }
        expected_physics_hz = 120
        expected_capture_hz = 10
        expected_capture_stride = 12
        expected_pour_durations = [1412, 1210, 1346]
        expected_pour_ends = [1412, 2622, 3968]
        expected_fill_end = 3968
        expected_primary_final_step = 4004
        expected_primary_final_offset_s = 0.3
        expected_no_inter_pour_settle = True
        segmented_stream_contract = False
        stream_segment_capture_counts = []
    expected_frame_count = len(expected_frame_steps)
    expected_video_fps = (
        F2_VIDEO_PLAYBACK_FPS
        if bound_runtime_plan
        else float(expected_capture_hz)
    )
    expected_video_repeat_plan = (
        build_time_preserving_video_repeat_plan(
            expected_frame_steps,
            physics_hz=expected_physics_hz,
            playback_fps=expected_video_fps,
        )
        if bound_runtime_plan
        else None
    )
    expected_liquid_representation = (
        COUNT_DRIVEN_RENDER_PATH if count_driven_body else "point_instancer"
    )
    if receipt.get("status") != "pass":
        failures.append("full_run_status_contract")
    captures = receipt.get("captures")
    required = (
        [
            item
            for item in captures
            if isinstance(item, Mapping)
            and isinstance(item.get("volume_gate"), Mapping)
            and item["volume_gate"].get("required") is True
        ]
        if isinstance(captures, list)
        else []
    )
    observed_volume_rows: list[tuple[int, float, int, int, int]] = []
    for item in required:
        gate = item["volume_gate"]
        try:
            observed_volume_rows.append(
                (
                    int(item["physics_step"]),
                    float(gate["programmed_cumulative_volume_ml"]),
                    int(gate["expected_cumulative_particle_count"]),
                    int(item["emitted_particle_count"]),
                    int(item["active_particle_count"]),
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            observed_volume_rows.append((-1, math.nan, -1, -1, -1))
    expected_volume_rows = [
        (step, volume_ml, count, count, count)
        for step, volume_ml, count in runtime_volume_rows
    ]
    if (
        observed_volume_rows != expected_volume_rows
        or any(
            item["volume_gate"].get("state_passed") is not True
            or not _combined_volume_gate_valid(item["volume_gate"])
            for item in required
        )
    ):
        failures.append("full_run_volume_contract")

    render = receipt.get("render")
    render = render if isinstance(render, Mapping) else {}
    frame_paths = render.get("frame_paths")
    frame_hashes = render.get("frame_sha256")
    frame_steps = render.get("frame_physics_steps")
    if (
        render.get("renderer") != "PathTracing"
        or render.get("resolution") != [640, 480]
        or render.get("rt_subframes") not in SUPPORTED_TWIN_RT_SUBFRAMES
        or render.get("pathtracing_total_spp_requested")
        not in SUPPORTED_TWIN_PATHTRACING_TOTAL_SPP
        or render.get("pathtracing_total_spp_readback")
        != render.get("pathtracing_total_spp_requested")
        or render.get("pathtracing_spp_requested")
        != render.get("pathtracing_total_spp_requested")
        or render.get("pathtracing_spp_readback")
        != render.get("pathtracing_spp_requested")
        or render.get("warmup_updates") != 60
        or render.get("frame_count") != expected_frame_count
        or not _artifact_manifest_valid(
            frame_paths, frame_hashes, count=expected_frame_count
        )
        or frame_steps != expected_frame_steps
    ):
        failures.append("full_run_rgb_contract")

    stage_indices = [
        expected_frame_steps.index(step) if step in expected_frame_steps else -1
        for step in stage_steps
    ]
    stage_paths = render.get("stage_frame_paths")
    stage_paths_valid = bool(
        isinstance(stage_paths, list)
        and len(stage_paths) == len(stage_indices)
        and all(isinstance(path, str) and path for path in stage_paths)
    )
    expected_stage_hashes = (
        [frame_hashes[index] for index in stage_indices]
        if (
            isinstance(frame_hashes, list)
            and len(frame_hashes) == expected_frame_count
            and all(index >= 0 for index in stage_indices)
        )
        else None
    )
    if (
        not stage_paths_valid
        or render.get("stage_frame_indices") != stage_indices
        or render.get("stage_frame_physics_steps")
        != stage_steps
        or render.get("stage_frame_sha256") != expected_stage_hashes
        or render.get("primary_final_frame_sha256")
        != (expected_stage_hashes[-1] if expected_stage_hashes else None)
        or render.get("stream_only_frame_count") != 0
        or (
            render.get(
                "count_driven_body_frame_count"
                if count_driven_body
                else "vial_proxy_frame_count"
            )
            != expected_frame_count
        )
        or (
            render.get(
                "count_driven_body_transition_step"
                if count_driven_body
                else "vial_proxy_transition_step"
            )
            != expected_frame_steps[0]
        )
        or render.get("liquid_representation_monotonic") is not True
    ):
        failures.append("full_run_stage_contract")

    video_path = render.get("video_path")
    if expected_video_repeat_plan is None:
        video_timing_valid = (
            render.get("video_decoded_frame_count") == expected_frame_count
        )
    else:
        video_duration_error = render.get("video_duration_error_s")
        video_timing_valid = bool(
            render.get("video_decoded_frame_count")
            == expected_video_repeat_plan["video_frame_count"]
            and render.get("video_playback_fps")
            == expected_video_repeat_plan["playback_fps"]
            and render.get("video_timing_algorithm")
            == expected_video_repeat_plan["algorithm"]
            and render.get("video_encoded_frame_time_rule")
            == expected_video_repeat_plan["encoded_frame_time_rule"]
            and render.get("video_timing_source_frame_steps")
            == expected_video_repeat_plan["source_frame_steps"]
            and render.get("video_source_capture_onset_frame_indices")
            == expected_video_repeat_plan[
                "source_capture_onset_frame_indices"
            ]
            and render.get("video_source_capture_onset_times_s")
            == expected_video_repeat_plan["source_capture_onset_times_s"]
            and render.get("video_source_capture_onset_errors_s")
            == expected_video_repeat_plan["source_capture_onset_errors_s"]
            and render.get("video_maximum_absolute_onset_error_s")
            == expected_video_repeat_plan["maximum_absolute_onset_error_s"]
            and render.get("video_frame_repeat_counts")
            == expected_video_repeat_plan["repeat_counts"]
            and render.get("video_frame_source_capture_indices")
            == expected_video_repeat_plan["source_capture_indices"]
            and _finite_number(render.get("video_source_duration_s"))
            and math.isclose(
                float(render["video_source_duration_s"]),
                float(expected_video_repeat_plan["source_duration_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and _finite_number(render.get("video_playback_duration_s"))
            and math.isclose(
                float(render["video_playback_duration_s"]),
                float(expected_video_repeat_plan["playback_duration_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and _finite_number(video_duration_error)
            and math.isclose(
                float(video_duration_error),
                float(expected_video_repeat_plan["duration_error_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and abs(float(video_duration_error)) <= 0.1
        )
    if (
        not isinstance(video_path, str)
        or not video_path
        or not _valid_sha256(render.get("video_sha256"))
        or not isinstance(render.get("video_codec"), str)
        or not render.get("video_codec")
        or not isinstance(render.get("video_container"), str)
        or not render.get("video_container")
        or not video_path.endswith(f".{render.get('video_container')}")
        or type(render.get("video_writer_fallback_used")) is not bool
        or render.get("video_decodable") is not True
        or not video_timing_valid
        or render.get("stage_frames_are_video_subsets") is not True
    ):
        failures.append("full_run_video_contract")

    timing = receipt.get("timing")
    timing = timing if isinstance(timing, Mapping) else {}
    physics_substeps = arguments.get(
        "physics_substeps",
        timing.get("physics_substeps_per_report_interval", 1),
    )
    if type(physics_substeps) is not int or physics_substeps not in {1, 2}:
        physics_substeps = -1
    expected_callback_count = runtime_final_step * physics_substeps
    expected_runtime_physics_hz = expected_physics_hz * physics_substeps
    two_substep_mode_valid = True
    if physics_substeps == 2:
        two_substep_mode_valid = _two_substep_receipt_mode_valid(
            receipt,
            arguments,
            runtime_final_step=runtime_final_step,
        )
    render_product_gating = render.get("render_product_update_gating")
    direct_physics_mode = bool(
        isinstance(render_product_gating, Mapping)
        and render_product_gating.get("mode")
        == "persistent_enabled_direct_physics"
    )
    if direct_physics_mode:
        physics_owner_valid = bool(
            physics_substeps == 1
            and timing.get("simulation_owner") == "SimulationManager.step"
            and timing.get("physics_step_method")
            == "SimulationManager.step(steps=1, update_fabric=False)"
            and timing.get("one_callback_per_direct_step") is True
            and timing.get("physics_loop_app_update_count") == 0
            and timing.get("capture_direct_physics_steps_advanced") == 0
            and timing.get("capture_simulation_time_advanced") is False
        )
    elif physics_substeps == 2:
        physics_owner_valid = _timeline_substep_timing_contract_valid(
            timing,
            expected_report_interval_count=runtime_final_step,
        )
    else:
        physics_owner_valid = timing.get("one_callback_per_app_update") is True
    if (
        arguments.get("resolution") != "640x480"
        or arguments.get("capture_stride") != expected_capture_stride
        or arguments.get("video_fps") != expected_video_fps
        or timing.get("physics_hz") != expected_runtime_physics_hz
        or timing.get("capture_hz") != expected_capture_hz
        or timing.get("pour_duration_steps") != expected_pour_durations
        or timing.get("pour_end_steps") != expected_pour_ends
        or timing.get("no_inter_pour_settle") is not expected_no_inter_pour_settle
        or timing.get("fill_end_step") != expected_fill_end
        or timing.get("primary_final_capture_step") != expected_primary_final_step
        or not _finite_number(timing.get("primary_final_offset_s"))
        or not math.isclose(
            float(timing.get("primary_final_offset_s", math.nan)),
            expected_primary_final_offset_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or timing.get("executed_physics_steps") != expected_callback_count
        or timing.get("callbacks_expected") != expected_callback_count
        or timing.get("callbacks_observed") != expected_callback_count
        or not physics_owner_valid
        or not two_substep_mode_valid
        or timing.get("capture_callbacks_advanced") != 0
        or timing.get("capture_timeline_time_advanced") is not False
        or timing.get("capture_playback_preserved") is not True
    ):
        failures.append("full_run_timing_contract")

    scene = receipt.get("scene_contract")
    scene = scene if isinstance(scene, Mapping) else {}
    hidden = scene.get("hidden_source_visibility")
    expected_hidden_paths = {
        "/FillStation/FillStation_LocalRoot/Fill_liquid",
        "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream",
    }
    hidden_valid = bool(
        isinstance(hidden, Mapping)
        and set(hidden) == expected_hidden_paths
        and all(
            isinstance(rows, Mapping)
            and rows
            and all(value == "invisible" for value in rows.values())
            for rows in hidden.values()
        )
    )
    source_layers = scene.get("source_layer_records")
    source_layers_valid = bool(
        isinstance(source_layers, list)
        and source_layers
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("resolved_path"), str)
            and row.get("resolved_path")
            and _valid_sha256(row.get("sha256"))
            for row in source_layers
        )
    )
    camera = receipt.get("camera")
    camera = camera if isinstance(camera, Mapping) else {}
    clip = camera.get("clipping_range_effective")
    clip_valid = bool(
        isinstance(clip, list)
        and len(clip) == 2
        and all(_finite_number(value) for value in clip)
        and math.isclose(
            float(clip[0]), 0.0007619999814778566, rel_tol=0.0, abs_tol=1.0e-9
        )
        and math.isclose(
            float(clip[1]), 25.399999618530273, rel_tol=0.0, abs_tol=1.0e-9
        )
    )
    parity = receipt.get("pre_pour_parity")
    parity = parity if isinstance(parity, Mapping) else {}
    parity_valid = bool(
        parity.get("passed") is True
        and all(
            isinstance(parity.get(path_key), str)
            and parity.get(path_key)
            and _valid_sha256(parity.get(hash_key))
            for path_key, hash_key in (
                ("reference_path", "reference_sha256"),
                ("f2_prefill_path", "f2_prefill_sha256"),
                ("absdiff_path", "absdiff_sha256"),
            )
        )
    )
    if (
        scene.get("default_prim") != "/FillStation"
        or scene.get("meters_per_unit") != 1.0
        or scene.get("up_axis") != "Z"
        or scene.get("source_layers_unchanged") is not True
        or not source_layers_valid
        or not hidden_valid
        or scene.get("forbidden_authored_paths") != []
        or scene.get("custom_camera_count") != 0
        or scene.get("custom_light_count") != 0
        or scene.get("custom_backdrop_count") != 0
        or camera.get("path") != PAIRED_CAPTURE_CAMERA_PATH
        or camera.get("source_has_authored_clipping_range") is not True
        or camera.get("clipping_range_authored_by_runner") is not False
        or not clip_valid
        or camera.get("resolution") != [640, 480]
        or not parity_valid
    ):
        failures.append("full_run_scene_contract")

    stream = receipt.get("stream_render")
    stream = stream if isinstance(stream, Mapping) else {}
    records = stream.get("records")
    stream_rows = []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                stream_rows.append(None)
                continue
            stream_rows.append(
                (
                    (
                        record.get("ordinal"),
                        record.get("segment_ordinal"),
                        record.get("segment_role"),
                        record.get("programmed_tank"),
                        record.get("tank_id"),
                        record.get("start_step"),
                        record.get("end_step"),
                    )
                    if segmented_stream_contract
                    else (
                        record.get("ordinal"),
                        record.get("tank_id"),
                        record.get("start_step"),
                        record.get("end_step"),
                    )
                )
            )
    if (
        stream.get("representation")
        != "render_only_mesh_current_tank_material"
        or stream_rows != expected_stream_rows
        or (
            segmented_stream_contract
            and (
                not stream_segment_capture_counts
                or min(stream_segment_capture_counts) < 2
            )
        )
        or not isinstance(records, list)
        or any(
            not isinstance(record, Mapping)
            or record.get("active_only_during_own_pour") is not True
            or (
                segmented_stream_contract
                and (
                    record.get("active_only_during_own_segment") is not True
                    or record.get("stream_and_birth_composition_equal") is not True
                )
            )
            or not isinstance(record.get("state_artifact_path"), str)
            or not record.get("state_artifact_path")
            or not _valid_sha256(record.get("state_artifact_sha256"))
            for record in records
        )
    ):
        failures.append("full_run_stream_contract")

    semantics = receipt.get("semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    pairs = semantics.get("pairs")
    mask_paths = [pair.get("mask_path") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    mask_hashes = [pair.get("mask_sha256") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    semantic_paths = [pair.get("semantic_path") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    semantic_hashes = [pair.get("semantic_sha256") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    instance_paths = [pair.get("instance_path") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    instance_hashes = [pair.get("instance_sha256") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    instance_mapping_paths = [pair.get("instance_mapping_path") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    instance_mapping_hashes = [pair.get("instance_mapping_sha256") for pair in pairs] if isinstance(pairs, list) and all(isinstance(pair, Mapping) for pair in pairs) else None
    pairs_valid = bool(
        isinstance(pairs, list)
        and len(pairs) == expected_frame_count
        and isinstance(frame_paths, list)
        and len(frame_paths) == expected_frame_count
        and isinstance(frame_hashes, list)
        and len(frame_hashes) == expected_frame_count
        and _artifact_manifest_valid(mask_paths, mask_hashes, count=expected_frame_count)
        and _artifact_manifest_valid(
            semantic_paths, semantic_hashes, count=expected_frame_count
        )
        and (
            not count_driven_body
            or (
                _artifact_manifest_valid(
                    instance_paths, instance_hashes, count=expected_frame_count
                )
                and _artifact_manifest_valid(
                    instance_mapping_paths,
                    instance_mapping_hashes,
                    count=expected_frame_count,
                )
            )
        )
        and all(
            pair.get("physics_step") == expected_frame_steps[index]
            and pair.get("rgb_path") == frame_paths[index]
            and pair.get("rgb_sha256") == frame_hashes[index]
            and pair.get("liquid_representation") == expected_liquid_representation
            and pair.get("shared_rgb_semantic_capture") is False
            and pair.get("paired_same_physics_state_rgb_semantic_capture") is True
            and pair.get("paired_capture_state_passed") is True
            and pair.get("mask_readable") is True
            and pair.get("mask_resolution") == [640, 480]
            and isinstance(pair.get("liquid_pixel_count"), int)
            and not isinstance(pair.get("liquid_pixel_count"), bool)
            and 0 < pair.get("liquid_pixel_count") < 640 * 480
            and pair.get("visible_stream_tanks")
            == expected_streams_by_step.get(expected_frame_steps[index])
            and (
                not count_driven_body
                or _stream_instance_semantic_pair_valid(
                    pair,
                    expected_streams_by_step.get(expected_frame_steps[index]),
                )
            )
            for index, pair in enumerate(pairs)
        )
    )
    if count_driven_body:
        try:
            duplicate_evidence = build_render_state_duplicate_evidence(receipt)
        except ValueError:
            duplicate_evidence = None
        if (
            not isinstance(duplicate_evidence, Mapping)
            or duplicate_evidence.get("passed") is not True
            or render.get("render_state_duplicate_evidence")
            != duplicate_evidence
        ):
            failures.append("full_run_render_state_frame_integrity")
    paired = semantics.get("paired_capture_contract")
    state_records = paired.get("state_records") if isinstance(paired, Mapping) else None
    annotation = semantics.get("transparent_annotation_override")
    transitions = annotation.get("transitions") if isinstance(annotation, Mapping) else None
    guards = timing.get("capture_guard_records")
    if (
        semantics.get("semantic_label") != "liquid"
        or semantics.get("semantic_type") != "class"
        or semantics.get("annotator") != "semantic_segmentation"
        or semantics.get("mask_source") != "exact_renderer_semantic_class_ids"
        or semantics.get("resolution") != [640, 480]
        or semantics.get(
            "count_driven_body_semantic_bound"
            if count_driven_body
            else "point_instancer_semantic_bound"
        )
        is not True
        or semantics.get("stream_semantic_bound") is not True
        or semantics.get("capture_volumes_enabled_readback") is not True
        or type(semantics.get("capture_volumes_restored_readback")) is not bool
        or semantics.get("semantic_gate_passed") is not True
        or not pairs_valid
        or semantics.get("frame_semantic_paths") != semantic_paths
        or semantics.get("frame_semantic_sha256") != semantic_hashes
        or semantics.get("stage_mask_sha256")
        != ([mask_hashes[index] for index in stage_indices] if mask_hashes else None)
        or not isinstance(state_records, list)
        or len(state_records) < expected_frame_count
        or not isinstance(guards, list)
        or len(guards) < 2 * expected_frame_count
        or not isinstance(transitions, list)
        or len(transitions) < 3 * expected_frame_count
    ):
        failures.append("full_run_semantic_contract")
    if semantics.get("training_eligible") is not False:
        failures.append("training_eligibility_contract")
    if count_driven_body:
        if isinstance(pairs, list) and any(
            not all(
                _valid_sha256(pair.get(field))
                for field in (
                    "body_points_sha256",
                    "body_extent_sha256",
                    "body_material_color_sha256",
                    "stream_visibility_sha256",
                    "compositions_sha256",
                )
            )
            or pair.get("body_visibility") != "inherited"
            for pair in pairs
            if isinstance(pair, Mapping)
        ):
            failures.append("count_driven_paired_geometry_contract")
        failures.extend(_validate_count_driven_body_contract(receipt))
    return tuple(dict.fromkeys(failures))


def _capture_guard_preserves_state(record: object, expected_pass: str) -> bool:
    if not isinstance(record, Mapping) or record.get("capture_pass") != expected_pass:
        return False
    callbacks_before = record.get("callbacks_before")
    callbacks_after = record.get("callbacks_after")
    time_before = record.get("timeline_time_before_s")
    time_after = record.get("timeline_time_after_s")
    return bool(
        isinstance(callbacks_before, int)
        and not isinstance(callbacks_before, bool)
        and callbacks_before == callbacks_after
        and _finite_number(time_before)
        and _finite_number(time_after)
        and math.isclose(
            float(time_before), float(time_after), rel_tol=0.0, abs_tol=1.0e-12
        )
        and type(record.get("timeline_playing_before")) is bool
        and record.get("timeline_playing_before")
        is record.get("timeline_playing_after")
    )


def _paired_state_record_valid(
    record: object,
    capture_guards: list[object],
    *,
    require_count_driven_body: bool = False,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    if record.get("passed") is not True or record.get("failed_predicates") != []:
        return False
    row_count = record.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        return False
    physics_step = record.get("physics_step")
    phase = record.get("phase")
    attempt = record.get("attempt")
    if (
        isinstance(physics_step, bool)
        or not isinstance(physics_step, int)
        or physics_step < 0
        or not isinstance(phase, str)
        or not phase
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt <= 0
    ):
        return False
    for field in ("callback_count", "timeline_time_s", "camera_path"):
        before = record.get(f"{field}_before")
        after = record.get(f"{field}_after")
        if before != after:
            return False
    if (
        not isinstance(record.get("callback_count_before"), int)
        or isinstance(record.get("callback_count_before"), bool)
        or not _finite_number(record.get("timeline_time_s_before"))
        or not isinstance(record.get("camera_path_before"), str)
        or record.get("camera_path_before") != PAIRED_CAPTURE_CAMERA_PATH
    ):
        return False
    for field in ("positions", "simulation_points", "velocities", "ids", "proto_indices"):
        before = record.get(f"{field}_sha256_before")
        after = record.get(f"{field}_sha256_after")
        if not _valid_sha256(before) or before != after:
            return False
    if require_count_driven_body:
        for field in (
            "compositions",
            "body_points",
            "body_extent",
            "body_visibility",
            "body_material_color",
            "stream_visibility",
        ):
            before = record.get(f"{field}_sha256_before")
            after = record.get(f"{field}_sha256_after")
            if not _valid_sha256(before) or before != after:
                return False
    rgb_index = record.get("rgb_guard_record_index")
    semantic_index = record.get("semantic_guard_record_index")
    if (
        isinstance(rgb_index, bool)
        or not isinstance(rgb_index, int)
        or isinstance(semantic_index, bool)
        or not isinstance(semantic_index, int)
        or rgb_index == semantic_index
        or not (0 <= rgb_index < len(capture_guards))
        or not (0 <= semantic_index < len(capture_guards))
    ):
        return False
    rgb_guard = capture_guards[rgb_index]
    semantic_guard = capture_guards[semantic_index]
    if not _capture_guard_preserves_state(
        rgb_guard, "kept_rgb_source_equivalent"
    ) or not _capture_guard_preserves_state(
        semantic_guard, "semantic_transparent_annotation"
    ):
        return False
    if (
        rgb_guard.get("callbacks_after")
        != semantic_guard.get("callbacks_before")
        or record.get("callback_count_before")
        != semantic_guard.get("callbacks_before")
        or record.get("callback_count_after")
        != semantic_guard.get("callbacks_after")
        or not math.isclose(
            float(rgb_guard.get("timeline_time_after_s")),
            float(semantic_guard.get("timeline_time_before_s")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(record.get("timeline_time_s_before")),
            float(semantic_guard.get("timeline_time_before_s")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isclose(
            float(record.get("timeline_time_s_after")),
            float(semantic_guard.get("timeline_time_after_s")),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        return False
    identity_fields = ("physics_step", "phase", "attempt")
    return all(
        record.get(field) == rgb_guard.get(field) == semantic_guard.get(field)
        for field in identity_fields
    )


def _copy_rows(value: object, name: str, width: int) -> list[list[float]]:
    rows = np.asarray(value, dtype=np.float64)
    if rows.size == 0:
        rows = np.empty((0, width), dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1:] != (width,) or not np.all(np.isfinite(rows)):
        raise ValueError(f"{name} must be finite with shape (N, {width})")
    return rows.tolist()


def append_point_instancer_state(
    previous: Mapping[str, object],
    *,
    positions: object,
    velocities: object,
    prototype_index: int,
    particle_ids: object,
) -> dict[str, object]:
    """Append one emission while preserving all PointInstancer row identity."""

    if isinstance(prototype_index, bool) or not isinstance(prototype_index, int) or prototype_index < 0:
        raise ValueError("prototype_index must be a non-negative integer")
    position_rows = _copy_rows(positions, "positions", 3)
    velocity_rows = _copy_rows(velocities, "velocities", 3)
    if len(position_rows) != len(velocity_rows) or not position_rows:
        raise ValueError("positions and velocities length mismatch or empty emission")
    ids = [int(value) for value in particle_ids]
    if len(ids) != len(position_rows) or len(set(ids)) != len(ids):
        raise ValueError("particle_ids length or uniqueness mismatch")
    existing_ids = [int(value) for value in previous.get("ids", [])]
    if len(set(existing_ids)) != len(existing_ids):
        raise ValueError("existing particle IDs are not stable")
    if existing_ids and ids[0] <= max(existing_ids):
        raise ValueError("new particle IDs must extend stable row identity")
    row_count = len(existing_ids)
    copied: dict[str, object] = {}
    widths = {"positions": 3, "velocities": 3, "orientations": 4, "scales": 3}
    for field in _ARRAY_FIELDS:
        rows = _copy_rows(previous.get(field, []), field, widths[field])
        if len(rows) != row_count:
            raise ValueError(f"{field} length drifted from stable IDs")
        copied[field] = rows
    simulation_rows = _copy_rows(
        previous.get("simulation_points", previous.get("positions", [])),
        "simulation_points",
        3,
    )
    if len(simulation_rows) != row_count:
        raise ValueError("simulation_points length drifted from stable IDs")
    copied["simulation_points"] = simulation_rows
    existing_proto = [int(value) for value in previous.get("proto_indices", [])]
    if len(existing_proto) != row_count:
        raise ValueError("proto_indices length drifted from stable IDs")
    copied["positions"].extend(position_rows)
    copied["simulation_points"].extend(position_rows)
    copied["velocities"].extend(velocity_rows)
    copied["orientations"].extend([[1.0, 0.0, 0.0, 0.0] for _ in position_rows])
    copied["scales"].extend([[1.0, 1.0, 1.0] for _ in position_rows])
    copied["ids"] = existing_ids + ids
    copied["proto_indices"] = existing_proto + [prototype_index] * len(position_rows)
    return copied


def append_runtime_particle_state(
    previous: Mapping[str, object],
    *,
    source_state: SourceState,
    positions: object,
    velocities: object,
) -> dict[str, object]:
    """Atomically extend all USD-facing and tracked composition row arrays."""

    if not isinstance(previous, Mapping):
        raise TypeError("previous runtime state must be a mapping")
    if not isinstance(source_state, SourceState):
        raise TypeError("source_state must be a SourceState")
    position_rows = _copy_rows(positions, "positions", 3)
    velocity_rows = _copy_rows(velocities, "velocities", 3)
    if not position_rows or len(position_rows) != len(velocity_rows):
        raise ValueError("positions and velocities length mismatch or empty emission")
    existing_ids = [int(value) for value in previous.get("ids", [])]
    new_ids = list(range(len(existing_ids), len(existing_ids) + len(position_rows)))

    result = append_point_instancer_state(
        previous,
        positions=position_rows,
        velocities=velocity_rows,
        prototype_index=source_state.palette_prototype_index,
        particle_ids=new_ids,
    )
    metadata_fields = {
        "scheduled_sources": source_state.scheduled_tank,
        "birth_compositions": list(source_state.birth_composition),
        "current_compositions": list(source_state.birth_composition),
        "birth_steps": source_state.birth_step,
        "birth_event_ids": source_state.birth_event_id,
        "pour_event_ids": source_state.pour_event_id,
    }
    for field, new_value in metadata_fields.items():
        existing = list(previous.get(field, []))
        if len(existing) != len(existing_ids):
            raise ValueError(f"{field} length drifted from stable IDs")
        result[field] = existing + [new_value for _ in position_rows]
    return result


_RUNTIME_SNAPSHOT_ARRAY_FIELDS = (
    "simulation_points",
    "positions",
    "velocities",
    "orientations",
    "scales",
    "ids",
    "proto_indices",
    "scheduled_sources",
    "birth_compositions",
    "current_compositions",
    "birth_steps",
    "birth_event_ids",
    "pour_event_ids",
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_npy_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def persist_runtime_particle_snapshot(
    output_dir: Path | str,
    event_id: str,
    state: Mapping[str, object],
    *,
    expected_ids: Sequence[int] | None = None,
    artifact_path_prefix: str | None = None,
) -> dict[str, object]:
    """Persist every runtime array before applying fail-closed assertions."""

    if not isinstance(event_id, str) or not event_id or Path(event_id).name != event_id:
        raise ValueError("event_id must be one safe path component")
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    root = Path(output_dir)
    if artifact_path_prefix is None:
        relative_base = Path(event_id)
    else:
        prefix = Path(artifact_path_prefix)
        if prefix.is_absolute() or ".." in prefix.parts or str(prefix) in {"", "."}:
            raise ValueError("artifact_path_prefix must be a safe relative directory")
        relative_base = prefix / event_id
    snapshot_dir = root / event_id
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, object]] = {}
    for name in _RUNTIME_SNAPSHOT_ARRAY_FIELDS:
        if name not in state:
            raise ValueError(f"runtime snapshot is missing {name}")
        if name in {"ids", "proto_indices", "birth_steps"}:
            dtype: object = np.int64
        elif name in {"scheduled_sources", "birth_event_ids", "pour_event_ids"}:
            dtype = np.str_
        else:
            dtype = np.float64
        array = np.ascontiguousarray(np.asarray(state[name], dtype=dtype))
        path = snapshot_dir / f"{name}.npy"
        _save_npy_atomic(path, array)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        arrays[name] = array
        records[name] = {
            "path": (relative_base / f"{name}.npy").as_posix(),
            "sha256": digest,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "finite": bool(array.dtype.kind not in "fc" or np.all(np.isfinite(array))),
        }
    snapshot_record = {
        "event_id": event_id,
        "arrays_saved_before_assertions": True,
        "arrays": records,
    }
    _atomic_json(snapshot_dir / "snapshot.json", snapshot_record)
    snapshot_record["snapshot_path"] = (relative_base / "snapshot.json").as_posix()
    snapshot_record["snapshot_sha256"] = hashlib.sha256(
        (snapshot_dir / "snapshot.json").read_bytes()
    ).hexdigest()

    failure: str | None = None
    row_count = len(arrays["ids"])
    if any(len(arrays[name]) != row_count for name in _RUNTIME_SNAPSHOT_ARRAY_FIELDS):
        failure = "runtime arrays lost stable row alignment"
    elif any(
        arrays[name].dtype.kind in "fc" and not np.all(np.isfinite(arrays[name]))
        for name in _RUNTIME_SNAPSHOT_ARRAY_FIELDS
    ):
        failure = "runtime arrays contain non-finite values"
    elif expected_ids is not None and arrays["ids"].tolist() != [
        int(value) for value in expected_ids
    ]:
        failure = "runtime arrays differ from stable particle IDs"
    if failure is not None:
        _atomic_json(
            snapshot_dir / "failure.json",
            {**snapshot_record, "failure": failure},
        )
        raise RuntimeError(failure)
    return snapshot_record


def build_runtime_capture_schedule(
    plan: Mapping[str, object],
    *,
    cap_delay_anchor: Mapping[str, object] | None = None,
    capture_stride_steps: int | None = None,
) -> dict[str, object]:
    """Build sparse review captures plus exact semantic timing anchors."""

    if not isinstance(plan, Mapping):
        raise TypeError("plan must be a mapping")
    timing = plan.get("timing")
    pours = plan.get("pours")
    if not isinstance(timing, Mapping) or not isinstance(pours, list) or not pours:
        raise ValueError("plan lacks timing or pours")
    physics_hz = timing.get("physics_hz")
    plan_capture_hz = timing.get("capture_hz")
    fill_end_step = timing.get("fill_end_step")
    if (
        isinstance(physics_hz, bool)
        or not isinstance(physics_hz, int)
        or physics_hz <= 0
        or isinstance(plan_capture_hz, bool)
        or not isinstance(plan_capture_hz, int)
        or plan_capture_hz <= 0
        or physics_hz % plan_capture_hz
        or isinstance(fill_end_step, bool)
        or not isinstance(fill_end_step, int)
        or fill_end_step <= 0
    ):
        raise ValueError("plan timing must contain integral positive rates and fill step")
    if capture_stride_steps is None:
        stride = physics_hz // plan_capture_hz
    else:
        if (
            isinstance(capture_stride_steps, bool)
            or not isinstance(capture_stride_steps, int)
            or capture_stride_steps <= 0
        ):
            raise ValueError("capture_stride_steps must be a positive integer")
        stride = capture_stride_steps
    capture_hz = physics_hz / stride
    captures: list[dict[str, object]] = []
    nonempty_segments: list[dict[str, object]] = []
    for pour in pours:
        if not isinstance(pour, Mapping):
            raise TypeError("each pour must be a mapping")
        start = int(pour["start_step"])
        end = int(pour["end_step"])
        ordinal = int(pour["ordinal"])
        tank = str(pour["tank_id"])
        raw_segments = pour.get("emission_segments")
        segments: list[dict[str, object]] = []
        if isinstance(raw_segments, list):
            for fallback_ordinal, segment in enumerate(raw_segments, start=1):
                if not isinstance(segment, Mapping):
                    raise TypeError("each emission segment must be a mapping")
                count = segment.get("particle_count")
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(
                        "emission segment particle_count must be a non-negative integer"
                    )
                if count == 0:
                    continue
                segment_start = segment.get("start_step")
                segment_end = segment.get("end_step")
                source = segment.get("source_tank_id")
                if (
                    isinstance(segment_start, bool)
                    or not isinstance(segment_start, int)
                    or isinstance(segment_end, bool)
                    or not isinstance(segment_end, int)
                    or not start <= segment_start <= segment_end <= end
                    or not isinstance(source, str)
                    or not source
                ):
                    raise ValueError(
                        "nonempty emission segment must have an in-pour step range "
                        "and source_tank_id"
                    )
                segment_row = {
                    "pour_ordinal": ordinal,
                    "programmed_tank": tank,
                    "segment_ordinal": int(
                        segment.get("ordinal", fallback_ordinal)
                    ),
                    "segment_role": str(
                        segment.get("segment_role", "current_tank")
                    ),
                    "source_tank_id": source,
                    "start_step": segment_start,
                    "end_step": segment_end,
                }
                segments.append(segment_row)
                nonempty_segments.append(segment_row)
        # Cadence is relative to the preceding pour boundary, not to the
        # one-indexed first physics step.  For example, the safe cadence samples
        # a pour beginning at step 1 at 312, 624, ... rather than 313, 625, ....
        boundary_step = start - 1
        semantic_anchor_steps = {
            step
            for left, right in zip(segments, segments[1:])
            for step in (int(left["end_step"]), int(right["start_step"]))
        }
        initial_motion_step = (
            F2_MOTION_FALSIFIER_STEP
            if ordinal == 1
            and segments
            and int(segments[0]["start_step"])
            <= F2_MOTION_FALSIFIER_STEP
            <= int(segments[0]["end_step"])
            else None
        )
        interior_step_set = {
            *range(boundary_step + stride, end, stride),
            *semantic_anchor_steps,
            *(() if initial_motion_step is None else (initial_motion_step,)),
        }
        segment_evidence_steps: set[int] = set()
        for segment in segments:
            segment_start = int(segment["start_step"])
            segment_end = int(segment["end_step"])
            captured = {
                step
                for step in interior_step_set
                if segment_start <= step <= segment_end
            }
            if len(captured) >= 2:
                continue
            midpoint = segment_start + (segment_end - segment_start) // 2
            for candidate in (midpoint, segment_end - 1, segment_start + 1):
                if (
                    len(captured) >= 2
                    or candidate < segment_start
                    or candidate > segment_end
                    or candidate >= end
                    or (
                        ordinal == 1
                        and candidate < F2_MOTION_FALSIFIER_STEP
                    )
                ):
                    continue
                interior_step_set.add(candidate)
                segment_evidence_steps.add(candidate)
                captured.add(candidate)
            if len(captured) < 2:
                raise ValueError(
                    "each nonempty emission segment must remain visible in at "
                    "least two captured pour frames after the motion falsifier"
                )
        interior_steps = sorted(interior_step_set)
        if len(interior_steps) < 3:
            raise ValueError(
                "each pour must expose first/middle/last review captures"
            )
        evidence_indices = {
            0: "first",
            len(interior_steps) // 2: "middle",
            len(interior_steps) - 1: "last",
        }
        for index, step in enumerate(interior_steps):
            source_tank = tank
            segment_metadata: dict[str, object] = {}
            if segments:
                matches = [
                    segment
                    for segment in segments
                    if int(segment["start_step"])
                    <= step
                    <= int(segment["end_step"])
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "each in-pour capture must belong to exactly one nonempty "
                        "emission segment"
                    )
                match = matches[0]
                source_tank = str(match["source_tank_id"])
                segment_metadata = {
                    "stream_source_tank": source_tank,
                    "source_segment_ordinal": int(match["segment_ordinal"]),
                    "source_segment_role": str(match["segment_role"]),
                }
            captures.append(
                {
                    "physics_step": step,
                    "kind": "in_pour",
                    "pour_ordinal": ordinal,
                    "scheduled_tank": tank,
                    "visible_stream_tanks": [source_tank],
                    "evidence_position": evidence_indices.get(index),
                    "semantic_anchor": (
                        "source_segment_boundary"
                        if step in semantic_anchor_steps
                        else "initial_stream_after_motion_falsifier"
                        if step == initial_motion_step
                        else "source_segment_evidence"
                        if step in segment_evidence_steps
                        else None
                    ),
                    **segment_metadata,
                }
            )
        final_boundary_visible = end == fill_end_step
        final_segment = segments[-1] if segments else None
        boundary_source = (
            str(final_segment["source_tank_id"])
            if final_segment is not None
            else tank
        )
        captures.append(
            {
                "physics_step": end,
                "kind": "stage_handoff",
                "pour_ordinal": ordinal,
                "scheduled_tank": tank,
                "visible_stream_tanks": (
                    [boundary_source] if final_boundary_visible else []
                ),
                "evidence_position": None,
                **(
                    {
                        "source_segment_ordinal": int(
                            final_segment["segment_ordinal"]
                        ),
                        "source_segment_role": str(
                            final_segment["segment_role"]
                        ),
                    }
                    if final_boundary_visible and final_segment is not None
                    else {}
                ),
            }
        )
    final_plus_0_3_step = fill_end_step + int(round(0.3 * physics_hz))
    final_plus_3_0_step = fill_end_step + int(round(3.0 * physics_hz))
    post_fill_end_step = final_plus_3_0_step
    for step in range(fill_end_step + stride, post_fill_end_step + 1, stride):
        captures.append(
            {
                "physics_step": step,
                "kind": (
                    "final_plus_0_3"
                    if step == final_plus_0_3_step
                    else "final_plus_3_0"
                    if step == final_plus_3_0_step
                    else "post_fill"
                ),
                "pour_ordinal": None,
                "scheduled_tank": None,
                "visible_stream_tanks": [],
                "evidence_position": None,
            }
        )
    captured_steps = {int(row["physics_step"]) for row in captures}
    for kind, step in (
        ("final_plus_0_3", final_plus_0_3_step),
        ("final_plus_3_0", final_plus_3_0_step),
    ):
        if step not in captured_steps:
            captures.append(
                {
                    "physics_step": step,
                    "kind": kind,
                    "pour_ordinal": None,
                    "scheduled_tank": None,
                    "visible_stream_tanks": [],
                    "evidence_position": None,
                }
            )
            captured_steps.add(step)
    cap_step: int | None = None
    if cap_delay_anchor is not None:
        digest = cap_delay_anchor.get("source_sha256")
        delay = cap_delay_anchor.get("delay_s")
        if (
            cap_delay_anchor.get("validated") is not True
            or not _valid_sha256(digest)
            or isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(float(delay))
            or float(delay) <= 0.0
        ):
            raise ValueError("cap-delay anchor must be validated, finite and hash-bound")
        cap_step = fill_end_step + int(round(float(delay) * physics_hz))
        matching = next(
            (row for row in captures if int(row["physics_step"]) == cap_step),
            None,
        )
        if matching is None:
            captures.append(
                {
                    "physics_step": cap_step,
                    "kind": "validated_cap_delay",
                    "pour_ordinal": None,
                    "scheduled_tank": None,
                    "visible_stream_tanks": [],
                    "evidence_position": None,
                    "validated_cap_delay": True,
                    "source_sha256": digest,
                }
            )
        else:
            matching["validated_cap_delay"] = True
            matching["source_sha256"] = digest
    for segment in nonempty_segments:
        visible_capture_steps = [
            int(row["physics_step"])
            for row in captures
            if row.get("kind") == "in_pour"
            and row.get("pour_ordinal") == segment["pour_ordinal"]
            and row.get("source_segment_ordinal")
            == segment["segment_ordinal"]
            and row.get("visible_stream_tanks")
            == [segment["source_tank_id"]]
        ]
        if len(visible_capture_steps) < 2:
            raise ValueError(
                "each nonempty emission segment must remain visible in at least "
                "two captured pour frames"
            )
    captures.sort(key=lambda row: (int(row["physics_step"]), str(row["kind"])))
    return {
        "physics_hz": physics_hz,
        "capture_hz": capture_hz,
        "capture_stride_steps": stride,
        "fill_end_step": fill_end_step,
        "final_plus_0_3_step": final_plus_0_3_step,
        "final_plus_3_0_step": final_plus_3_0_step,
        "validated_cap_delay_step": cap_step,
        "captures": captures,
    }


def build_time_preserving_video_repeat_plan(
    frame_physics_steps: object,
    *,
    physics_hz: int,
    playback_fps: float,
) -> dict[str, object]:
    """Map sparse source captures to a fixed-rate playback timeline.

    Each source capture is held until the next source capture (zero-order hold).
    Onsets come from absolute physics timestamps rather than independently
    rounded gaps, so rounding cannot drift.  When adjacent semantic captures
    quantize to the same playback frame, the earlier capture is moved to the
    immediately preceding frame; every source state remains represented.
    """

    if not isinstance(frame_physics_steps, (list, tuple)):
        raise TypeError("frame_physics_steps must be a list or tuple")
    if not frame_physics_steps:
        raise ValueError("frame_physics_steps must not be empty")
    if (
        isinstance(physics_hz, bool)
        or not isinstance(physics_hz, int)
        or physics_hz <= 0
    ):
        raise ValueError("physics_hz must be a positive integer")
    if (
        isinstance(playback_fps, bool)
        or not isinstance(playback_fps, (int, float))
        or not math.isfinite(float(playback_fps))
        or float(playback_fps) <= 0.0
    ):
        raise ValueError("playback_fps must be a finite positive number")
    steps: list[int] = []
    previous = 0
    for raw_step in frame_physics_steps:
        if (
            isinstance(raw_step, bool)
            or not isinstance(raw_step, int)
            or raw_step <= previous
        ):
            raise ValueError(
                "frame_physics_steps must be strictly increasing positive integers"
            )
        steps.append(raw_step)
        previous = raw_step

    fps = float(playback_fps)
    source_times_s = [step / physics_hz for step in steps]
    ideal_onset_indices = [time_s * fps for time_s in source_times_s]
    target_frame_count = math.floor(steps[-1] * fps / physics_hz + 0.5)
    if target_frame_count < len(steps):
        raise ValueError(
            "playback_fps is too low to represent every sparse source capture"
        )

    # Nominal zero-order-hold onset is the first encoded-frame interval whose
    # midpoint is at or after the absolute capture timestamp.  Resolve
    # collisions backwards, keeping the later state at its nominal interval
    # and placing every earlier state in the nearest available preceding one.
    # The final sample index is one less than the encoded-frame denominator
    # because a video frame occupies an interval rather than a zero-duration
    # point.
    onset_indices = [
        int(math.ceil(value - 0.5 - 1.0e-12))
        for value in ideal_onset_indices
    ]
    onset_indices[-1] = min(onset_indices[-1], target_frame_count - 1)
    for index in range(len(onset_indices) - 2, -1, -1):
        onset_indices[index] = min(
            onset_indices[index], onset_indices[index + 1] - 1
        )
    onset_indices[0] = 0
    if any(
        right <= left for left, right in zip(onset_indices, onset_indices[1:])
    ):
        raise ValueError(
            "playback_fps is too low to preserve ordered sparse source captures"
        )
    repeat_counts = [
        right - left for left, right in zip(onset_indices, onset_indices[1:])
    ] + [target_frame_count - onset_indices[-1]]
    if any(count <= 0 for count in repeat_counts):
        raise RuntimeError("zero-order-hold video plan omitted a source capture")

    source_capture_indices = [
        capture_index
        for capture_index, repeat_count in enumerate(repeat_counts)
        for _ in range(repeat_count)
    ]
    source_duration_s = steps[-1] / physics_hz
    playback_duration_s = len(source_capture_indices) / fps
    onset_times_s = [(index + 0.5) / fps for index in onset_indices]
    onset_errors_s = [
        onset_time - source_time
        for onset_time, source_time in zip(onset_times_s, source_times_s)
    ]
    return {
        "algorithm": "absolute_midpoint_zero_order_hold_collision_preserving_v1",
        "encoded_frame_time_rule": "interval_midpoint_(frame_index_plus_0.5)_div_playback_fps",
        "source_frame_steps": steps,
        "physics_hz": physics_hz,
        "playback_fps": fps,
        "source_capture_onset_frame_indices": onset_indices,
        "source_capture_onset_times_s": onset_times_s,
        "source_capture_onset_errors_s": onset_errors_s,
        "maximum_absolute_onset_error_s": max(map(abs, onset_errors_s)),
        "repeat_counts": repeat_counts,
        "source_capture_indices": source_capture_indices,
        "video_frame_count": len(source_capture_indices),
        "source_duration_s": source_duration_s,
        "playback_duration_s": playback_duration_s,
        "duration_error_s": playback_duration_s - source_duration_s,
    }


def robust_contained_surface_z_m(
    positions: object,
    *,
    bottom_z_m: float,
    mouth_z_m: float,
    render_radius_m: float,
) -> float:
    """Estimate the contained surface without following a narrow falling tail."""

    rows = np.asarray(positions)
    if rows.size == 0:
        return float(bottom_z_m)
    if rows.dtype.kind not in "fiu":
        raise TypeError("positions must contain real numbers")
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1:] != (3,) or not np.all(np.isfinite(rows)):
        raise ValueError("positions must be finite with shape (N, 3)")
    values = []
    for name, value in (
        ("bottom_z_m", bottom_z_m),
        ("mouth_z_m", mouth_z_m),
        ("render_radius_m", render_radius_m),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        values.append(number)
    bottom, mouth, radius = values
    if mouth <= bottom or radius <= 0.0:
        raise ValueError("mouth must exceed bottom and render radius must be positive")
    contained = rows[(rows[:, 2] >= bottom) & (rows[:, 2] <= mouth), 2]
    if contained.size == 0:
        return bottom
    q95 = float(np.quantile(contained, 0.95, method="linear"))
    return min(max(q95 + radius, bottom), mouth)


def bind_external_nuisance_receipt(
    path: Path | str, expected_sha256: str
) -> dict[str, object]:
    """Bind measured nuisance evidence without changing any physics setting."""

    source = Path(path)
    if not source.is_absolute() or not source.is_file():
        raise ValueError("nuisance receipt must be an existing absolute file")
    if not _valid_sha256(expected_sha256):
        raise ValueError("expected nuisance receipt SHA-256 is invalid")
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError("nuisance receipt SHA-256 mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("nuisance receipt is unreadable") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("receipt_type") != "real_photometric_nuisance_range"
        or payload.get("status") != "measured"
        or payload.get("photometric_status") != "measured"
        or payload.get("shadow_status") != "measured"
    ):
        raise ValueError("nuisance receipt is not a measured photometric receipt")
    pooled = payload.get("pooled")
    if not isinstance(pooled, Mapping):
        raise ValueError("nuisance receipt pooled evidence is missing")
    for field in ("candidate_count", "usable_count", "shadow_observation_count"):
        value = pooled.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"nuisance receipt {field} is invalid")
    bounds = pooled.get("percentile_bounds")
    if not isinstance(bounds, Mapping) or not bounds:
        raise ValueError("nuisance receipt percentile bounds are missing")
    return {
        "receipt_path": str(source),
        "receipt_sha256": actual,
        "status": "measured",
        "usable_count": int(pooled["usable_count"]),
        "candidate_count": int(pooled["candidate_count"]),
        "shadow_observation_count": int(pooled["shadow_observation_count"]),
        "percentile_bounds": json.loads(
            json.dumps(bounds, sort_keys=True, allow_nan=False)
        ),
        "physics_parameters_modified": False,
        "authoring_status": "pending_task5_runtime_readback",
    }


def build_f2_runtime_acceptance_receipt(
    *,
    runtime_binding: Mapping[str, object],
    artifact_validation: Mapping[str, object],
    gate_evidence: Mapping[str, object],
) -> dict[str, object]:
    """Build Task 3's small downstream receipt after fail-closed validation."""

    required_runtime_keys = {
        "provider",
        "runtime_name",
        "isaac_sim_version",
        "runtime_id",
        "source_sha256",
    }
    if set(runtime_binding) != required_runtime_keys:
        raise ValueError("runtime binding has unexpected fields")
    if (
        runtime_binding.get("provider") != "brev"
        or runtime_binding.get("runtime_name") != "NVIDIA Isaac Sim"
        or runtime_binding.get("isaac_sim_version") != "6.2.0.0"
        or not isinstance(runtime_binding.get("runtime_id"), str)
        or not runtime_binding["runtime_id"]
        or not _valid_sha256(runtime_binding.get("source_sha256"))
    ):
        raise ValueError("runtime binding is not the accepted F2 runtime")
    failures = artifact_validation.get("failures")
    if (
        artifact_validation.get("status") != "pass"
        or not isinstance(failures, list)
        or failures
        or isinstance(artifact_validation.get("verified_artifact_count"), bool)
        or not isinstance(artifact_validation.get("verified_artifact_count"), int)
        or artifact_validation["verified_artifact_count"] <= 0
    ):
        raise ValueError("underlying F2 artifacts have not passed validation")
    if set(gate_evidence) != F2_REQUIRED_ACCEPTANCE_GATES:
        raise ValueError("acceptance gate evidence is incomplete")
    failed = sorted(name for name, passed in gate_evidence.items() if passed is not True)
    if failed:
        raise ValueError(f"F2 acceptance gates failed: {failed!r}")
    return {
        "schema_version": 1,
        "artifact_type": "f2_runtime_acceptance_receipt",
        "status": "accepted",
        "provider": "brev",
        "runtime_id": runtime_binding["runtime_id"],
        "isaac_sim_version": "6.2.0.0",
        "source_sha256": runtime_binding["source_sha256"],
        "gates": {name: True for name in sorted(F2_REQUIRED_ACCEPTANCE_GATES)},
    }


def build_prototype_specs(palette: Mapping[str, Sequence[float]]) -> tuple[dict[str, object], ...]:
    """Describe one semantic liquid sphere prototype per fixed colour."""

    specs: list[dict[str, object]] = []
    for index, (tank_id, color) in enumerate(palette.items()):
        values = [float(value) for value in color]
        if len(values) != 3 or any(not math.isfinite(value) for value in values):
            raise ValueError("prototype colour must contain three finite values")
        specs.append(
            {
                "prototype_index": index,
                "tank_id": str(tank_id),
                "shape": "Sphere",
                "semantic_label": "liquid",
                "color": values,
            }
        )
    if not specs:
        raise ValueError("palette must not be empty")
    return tuple(specs)


def build_palette_material_path(namespace: str, prototype_index: int) -> str:
    """Return an absolute USD material path with a valid named prim leaf."""

    if not isinstance(namespace, str) or not namespace.startswith("/") or namespace.endswith("/"):
        raise ValueError("namespace must be a non-root absolute USD prim path")
    if isinstance(prototype_index, bool) or not isinstance(prototype_index, int) or prototype_index < 0:
        raise ValueError("prototype_index must be a non-negative integer")
    return f"{namespace}/PaletteMaterials/Palette_{prototype_index:02d}"


def containment_evidence(
    positions: object,
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    expected_in_flight_mask: object | None = None,
    cylindrical_center_xy: Sequence[float] | None = None,
    inner_radius_m: float | None = None,
    polygon_center_xy: Sequence[float] | None = None,
    polygon_apothem_m: float | None = None,
    polygon_wall_count: int | None = None,
    numerical_tolerance_m: float = 1.0e-6,
) -> dict[str, object]:
    """Count particles inside the fitted vial interior and expose leakage."""

    rows = np.asarray(positions, dtype=np.float64)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1:] != (3,) or not np.all(np.isfinite(rows)):
        raise ValueError("positions must be finite with shape (N, 3)")
    if lower.shape != (3,) or upper.shape != (3,) or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds must contain three finite values")
    if np.any(upper <= lower):
        raise ValueError("upper bounds must exceed lower bounds")
    if numerical_tolerance_m < 0.0 or not math.isfinite(float(numerical_tolerance_m)):
        raise ValueError("numerical_tolerance_m must be finite and non-negative")
    inside_z = np.all(
        (rows[:, 2:] >= lower[None, 2:] - numerical_tolerance_m)
        & (rows[:, 2:] <= upper[None, 2:] + numerical_tolerance_m),
        axis=1,
    )
    conservative_inside: np.ndarray | None = None
    if cylindrical_center_xy is not None or inner_radius_m is not None:
        if cylindrical_center_xy is None or inner_radius_m is None:
            raise ValueError("cylindrical center and radius must be supplied together")
        center = np.asarray(cylindrical_center_xy, dtype=np.float64)
        radius = float(inner_radius_m)
        if center.shape != (2,) or not np.all(np.isfinite(center)) or not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("cylindrical center/radius must be finite and valid")
        conservative_inside = (
            np.sum((rows[:, :2] - center[None, :]) ** 2, axis=1)
            <= (radius + numerical_tolerance_m) ** 2
        ) & inside_z
    else:
        center = None
        radius = None
    polygon_values = (polygon_center_xy, polygon_apothem_m, polygon_wall_count)
    if any(value is not None for value in polygon_values):
        if any(value is None for value in polygon_values):
            raise ValueError("polygon center, apothem and wall count must be supplied together")
        polygon_center = np.asarray(polygon_center_xy, dtype=np.float64)
        apothem = float(polygon_apothem_m)
        if (
            polygon_center.shape != (2,)
            or not np.all(np.isfinite(polygon_center))
            or not math.isfinite(apothem)
            or apothem <= 0.0
            or isinstance(polygon_wall_count, bool)
            or not isinstance(polygon_wall_count, int)
            or polygon_wall_count < 3
        ):
            raise ValueError("polygon center/apothem/wall count must be finite and valid")
        angles = 2.0 * math.pi * np.arange(polygon_wall_count) / polygon_wall_count
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        centered = rows[:, :2] - polygon_center[None, :]
        inside_xy = np.all(
            centered @ normals.T <= apothem + numerical_tolerance_m,
            axis=1,
        )
        inside = inside_xy & inside_z
    elif conservative_inside is not None:
        polygon_center = None
        apothem = None
        inside = conservative_inside
    else:
        polygon_center = None
        apothem = None
        inside = np.all((rows >= lower[None, :]) & (rows <= upper[None, :]), axis=1)
    if expected_in_flight_mask is None:
        in_flight = np.zeros(len(rows), dtype=bool)
    else:
        in_flight = np.asarray(expected_in_flight_mask, dtype=bool)
        if in_flight.shape != (len(rows),):
            raise ValueError("expected_in_flight_mask length must match positions")
        if np.any(in_flight & inside):
            raise ValueError("in-flight particles cannot also be inside the vial")
    outside = ~inside
    leaked = outside & ~in_flight
    max_abs = np.max(np.abs(rows), axis=0) if len(rows) else np.zeros(3)
    contained_z = rows[inside, 2]
    if len(contained_z):
        z_q05, z_q95 = np.quantile(contained_z, (0.05, 0.95), method="linear")
        z_min: float | None = float(np.min(contained_z))
        z_max: float | None = float(np.max(contained_z))
        z_q05_value: float | None = float(z_q05)
        z_q95_value: float | None = float(z_q95)
    else:
        z_min = z_max = z_q05_value = z_q95_value = None
    return {
        "active_particle_count": int(len(rows)),
        "inside_vial_interior_count": int(np.count_nonzero(inside)),
        "outside_vial_interior_count": int(np.count_nonzero(outside)),
        "expected_in_flight_stream_count": int(np.count_nonzero(in_flight)),
        "leaked_particle_count": int(np.count_nonzero(leaked)),
        "max_abs_bounds_m": [float(value) for value in max_abs],
        "lower_bounds_m": lower.tolist(),
        "upper_bounds_m": upper.tolist(),
        "cylindrical_center_xy_m": None if cylindrical_center_xy is None else [float(value) for value in cylindrical_center_xy],
        "inner_radius_m": None if inner_radius_m is None else float(inner_radius_m),
        "inside_conservative_circle_count": (
            None
            if conservative_inside is None
            else int(np.count_nonzero(conservative_inside))
        ),
        "exact_polygon_containment": polygon_center_xy is not None,
        "polygon_center_xy_m": (
            None if polygon_center is None else [float(value) for value in polygon_center]
        ),
        "polygon_apothem_m": None if apothem is None else float(apothem),
        "polygon_wall_count": polygon_wall_count,
        "finite_state": True,
        "contained_z_min_m": z_min,
        "contained_z_q05_m": z_q05_value,
        "contained_z_q95_m": z_q95_value,
        "contained_z_max_m": z_max,
        "numerical_tolerance_m": float(numerical_tolerance_m),
        "leakage_detected": bool(np.any(leaked)),
    }


def build_capture_record(
    *,
    physics_step: int,
    phase: str,
    containment: Mapping[str, object],
    prototype_count: int,
    instancer_path: str,
    emitted_particle_count: int | None = None,
    programmed_cumulative_volume_ml: float | None = None,
    expected_cumulative_particle_count: int | None = None,
    cross_section_area_m2: float | None = None,
    height_tolerance_m: float | None = None,
    render_radius_m: float | None = None,
    point_instancer_scales: object | None = None,
    smoothed_render_containment: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if isinstance(physics_step, bool) or not isinstance(physics_step, int) or physics_step < 0:
        raise ValueError("physics_step must be a non-negative integer")
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be non-empty")
    if isinstance(prototype_count, bool) or not isinstance(prototype_count, int) or prototype_count <= 0:
        raise ValueError("prototype_count must be positive")
    if not isinstance(instancer_path, str) or not instancer_path:
        raise ValueError("instancer_path must be non-empty")
    required = {
        "active_particle_count",
        "inside_vial_interior_count",
        "outside_vial_interior_count",
        "expected_in_flight_stream_count",
        "leaked_particle_count",
        "max_abs_bounds_m",
        "leakage_detected",
        "inside_conservative_circle_count",
        "exact_polygon_containment",
        "finite_state",
        "contained_z_min_m",
        "contained_z_q05_m",
        "contained_z_q95_m",
        "contained_z_max_m",
    }
    if not required.issubset(containment):
        raise ValueError("containment evidence is incomplete")
    active_count = int(containment["active_particle_count"])
    emitted_count = active_count if emitted_particle_count is None else emitted_particle_count
    if (
        isinstance(emitted_count, bool)
        or not isinstance(emitted_count, int)
        or emitted_count < 0
    ):
        raise ValueError("emitted_particle_count must be a non-negative integer")
    stage_values = (
        programmed_cumulative_volume_ml,
        expected_cumulative_particle_count,
        cross_section_area_m2,
        height_tolerance_m,
        render_radius_m,
        point_instancer_scales,
        smoothed_render_containment,
    )
    volume_gate_required = any(value is not None for value in stage_values)
    if volume_gate_required and any(value is None for value in stage_values):
        raise ValueError("stage volume evidence inputs must be supplied together")
    scale_min: list[float] | None = None
    scale_max: list[float] | None = None
    volume_gate: dict[str, object] = {
        "required": volume_gate_required,
        "passed": None,
        "failure_reasons": [],
    }
    if volume_gate_required:
        if (
            isinstance(expected_cumulative_particle_count, bool)
            or not isinstance(expected_cumulative_particle_count, int)
            or expected_cumulative_particle_count <= 0
        ):
            raise ValueError("expected_cumulative_particle_count must be positive")
        if not isinstance(smoothed_render_containment, Mapping):
            raise TypeError("smoothed_render_containment must be a mapping")
        scales = np.asarray(point_instancer_scales)
        if scales.dtype.kind not in "fiu":
            raise TypeError("point_instancer_scales must contain real numbers")
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (active_count, 3) or not np.all(np.isfinite(scales)):
            raise ValueError("point_instancer_scales must be finite with shape (N, 3)")
        scale_min = [float(value) for value in np.min(scales, axis=0)]
        scale_max = [float(value) for value in np.max(scales, axis=0)]
        try:
            expected_height = height_for_volume_ml(programmed_cumulative_volume_ml)
            height_lower_bound, height_upper_bound = height_bounds_for_volume_ml(
                programmed_cumulative_volume_ml
            )
            height_gate_model = "geometry_curve_h_95_105_volume"
            curve_sha256: str | None = volume_height_curve_digest()
        except ValueError as exc:
            # Read historical synthetic unit fixtures whose volumes exceed the
            # physical vial curve. Production F2 targets are all curve-bounded.
            if "exceeds curve capacity" not in str(exc):
                raise
            expected_height = expected_cylindrical_height_m(
                programmed_cumulative_volume_ml, cross_section_area_m2
            )
            height_lower_bound = expected_height - float(height_tolerance_m)
            height_upper_bound = expected_height + float(height_tolerance_m)
            height_gate_model = "legacy_cylinder_tolerance_outside_curve_domain"
            curve_sha256 = None
        q05 = containment["contained_z_q05_m"]
        q95 = containment["contained_z_q95_m"]
        particle_height = (
            particle_bulk_height_from_quantiles_m(q05, q95, render_radius_m)
            if q05 is not None and q95 is not None
            else None
        )
        render_q05 = smoothed_render_containment.get("contained_z_q05_m")
        render_q95 = smoothed_render_containment.get("contained_z_q95_m")
        smoothed_render_height = (
            particle_bulk_height_from_quantiles_m(
                render_q05,
                render_q95,
                render_radius_m,
            )
            if render_q05 is not None and render_q95 is not None
            else None
        )
        failures: list[str] = []
        if containment["exact_polygon_containment"] is not True:
            failures.append("exact_dodecagon_containment_missing")
        if containment["finite_state"] is not True:
            failures.append("particle_state_nonfinite")
        if emitted_count != expected_cumulative_particle_count:
            failures.append("emitted_count_mismatch")
        if active_count != emitted_count:
            failures.append("active_count_mismatch")
        if int(containment["inside_vial_interior_count"]) != emitted_count:
            failures.append("inside_count_mismatch")
        if int(containment["expected_in_flight_stream_count"]) != 0:
            failures.append("in_flight_particles_at_stage")
        if int(containment["leaked_particle_count"]) != 0:
            failures.append("outside_or_escaped_particles")
        if (
            particle_height is None
            or not height_lower_bound <= particle_height <= height_upper_bound
        ):
            failures.append("particle_height_outside_geometry_volume_bounds")
        volume_gate = {
            "required": True,
            "state_passed": not failures,
            "visible_passed": None,
            "passed": False,
            "failure_reasons": failures,
            "programmed_cumulative_volume_ml": float(programmed_cumulative_volume_ml),
            "expected_cumulative_particle_count": expected_cumulative_particle_count,
            "cross_section_area_m2": float(cross_section_area_m2),
            "expected_height_m": expected_height,
            "height_gate_model": height_gate_model,
            "volume_height_curve_sha256": curve_sha256,
            "height_lower_bound_m": height_lower_bound,
            "height_upper_bound_m": height_upper_bound,
            "particle_bulk_height_m": particle_height,
            "visible_height_policy": {
                "policy_id": "real_video5_opaque_base_cutoff_v1",
                "required": float(programmed_cumulative_volume_ml)
                > OPAQUE_BASE_CUTOFF_VOLUME_ML,
                "opaque_base_cutoff_volume_ml": OPAQUE_BASE_CUTOFF_VOLUME_ML,
                "opaque_base_cutoff_height_m": OPAQUE_BASE_CUTOFF_HEIGHT_M,
                "expected_height_above_cutoff_m": max(
                    0.0, expected_height - OPAQUE_BASE_CUTOFF_HEIGHT_M
                ),
                "lower_bound_height_above_cutoff_m": max(
                    0.0, height_lower_bound - OPAQUE_BASE_CUTOFF_HEIGHT_M
                ),
                "upper_bound_height_above_cutoff_m": max(
                    0.0, height_upper_bound - OPAQUE_BASE_CUTOFF_HEIGHT_M
                ),
                "comparison_space": "liquid_height_above_opaque_base_cutoff_m",
                "source": OPAQUE_BASE_CUTOFF_SOURCE,
                "calibrated": False,
            },
            # Retained only to read historical receipts; the geometry curve is
            # authoritative for new runtime decisions.
            "height_tolerance_m": float(height_tolerance_m),
            "render_radius_m": float(render_radius_m),
            "nominal_particle_volume_ml": float(
                8.0 * float(render_radius_m) ** 3 * 1_000_000.0
            ),
            "particle_count_volume_formula": "(2 * fluid_rest_offset_m)^3",
            "particle_count_volume_source": (
                "NVIDIA FluidBallEmitterDemo.py: particleSpacing = "
                "2 * fluidRestOffset"
            ),
            "expected_count_nominal_volume_ml": float(
                expected_cumulative_particle_count
                * 8.0
                * float(render_radius_m) ** 3
                * 1_000_000.0
            ),
            "smoothed_render_state": {
                "inside_vial_interior_count": smoothed_render_containment.get(
                    "inside_vial_interior_count"
                ),
                "contained_z_min_m": smoothed_render_containment.get(
                    "contained_z_min_m"
                ),
                "contained_z_q05_m": render_q05,
                "contained_z_q95_m": render_q95,
                "contained_z_max_m": smoothed_render_containment.get(
                    "contained_z_max_m"
                ),
                "particle_bulk_height_m": smoothed_render_height,
            },
        }
    return {
        "physics_step": physics_step,
        "phase": phase,
        "emitted_particle_count": emitted_count,
        "active_particle_count": containment["active_particle_count"],
        "inside_vial_interior_count": containment["inside_vial_interior_count"],
        "outside_vial_interior_count": containment["outside_vial_interior_count"],
        "expected_in_flight_stream_count": containment["expected_in_flight_stream_count"],
        "in_flight_above_mouth_count": containment["expected_in_flight_stream_count"],
        "leaked_particle_count": containment["leaked_particle_count"],
        "outside_or_escaped_particle_count": containment["leaked_particle_count"],
        "inside_conservative_circle_count": containment["inside_conservative_circle_count"],
        "exact_polygon_containment": containment["exact_polygon_containment"],
        "finite_state": containment["finite_state"],
        "contained_z_min_m": containment["contained_z_min_m"],
        "contained_z_q05_m": containment["contained_z_q05_m"],
        "contained_z_q95_m": containment["contained_z_q95_m"],
        "contained_z_max_m": containment["contained_z_max_m"],
        "point_instancer_scale_component_min": scale_min,
        "point_instancer_scale_component_max": scale_max,
        "volume_gate": volume_gate,
        "max_abs_bounds_m": list(containment["max_abs_bounds_m"]),
        "leakage_detected": containment["leakage_detected"],
        "prototype_count": prototype_count,
        "instancer_path": instancer_path,
        "max_velocity_m_s": MAX_VELOCITY_M_S,
        "render_limitations": dict(RENDER_LIMITATIONS),
    }


def attach_visible_volume_evidence(
    capture_record: Mapping[str, object],
    liquid_mask: object,
    *,
    center_xy_m: tuple[float, float],
    bottom_z_m: float,
    mouth_z_m: float,
    circumradius_m: float,
) -> dict[str, object]:
    """Attach or explicitly skip the opaque-base-aware visible-height gate."""

    if not isinstance(capture_record, Mapping):
        raise TypeError("capture_record must be a mapping")
    result = dict(capture_record)
    existing_gate = capture_record.get("volume_gate")
    if not isinstance(existing_gate, Mapping):
        raise ValueError("capture_record has no volume gate")
    gate = dict(existing_gate)
    result["volume_gate"] = gate
    if gate.get("required") is not True:
        gate.update(
            {
                "visible_passed": None,
                "visible_mask_evidence": {"required": False},
                "passed": gate.get("state_passed") is True,
            }
        )
        return result
    visible_policy = gate.get("visible_height_policy")
    if isinstance(visible_policy, Mapping) and visible_policy.get("required") is False:
        gate.update(
            {
                "visible_passed": None,
                "visible_mask_evidence": {
                    "required": False,
                    "mask_evaluated": False,
                    "status": "not_required_at_or_below_opaque_base_cutoff",
                    "programmed_cumulative_volume_ml": gate.get(
                        "programmed_cumulative_volume_ml"
                    ),
                    "opaque_base_cutoff_volume_ml": visible_policy.get(
                        "opaque_base_cutoff_volume_ml"
                    ),
                    "opaque_base_cutoff_height_m": visible_policy.get(
                        "opaque_base_cutoff_height_m"
                    ),
                    "source": visible_policy.get("source"),
                },
                "passed": gate.get("state_passed") is True,
            }
        )
        return result
    mask = np.asarray(liquid_mask)
    if mask.ndim != 2:
        raise ValueError("liquid_mask must be two-dimensional")
    roi = projected_vial_geometry_roi_mask(
        mask.shape,
        center_xy_m=center_xy_m,
        bottom_z_m=bottom_z_m,
        mouth_z_m=mouth_z_m,
        circumradius_m=circumradius_m,
    )
    failures = list(gate.get("failure_reasons", []))
    try:
        surface = find_dense_surface_row(mask, geometry_roi_mask=roi)
    except (TypeError, ValueError) as error:
        visible_evidence: dict[str, object] = {
            "required": True,
            "passed": False,
            "status": "missing_dense_surface",
            "error": str(error),
        }
        visible_passed = False
    else:
        visible_height = visible_height_from_surface_row_m(
            surface["surface_row_px"],
            center_xy_m=center_xy_m,
            bottom_z_m=bottom_z_m,
            circumradius_m=circumradius_m,
            maximum_height_m=float(mouth_z_m) - float(bottom_z_m),
        )
        cutoff_height = (
            float(visible_policy["opaque_base_cutoff_height_m"])
            if isinstance(visible_policy, Mapping)
            else 0.0
        )
        visible_height_above_cutoff = max(0.0, visible_height - cutoff_height)
        lower_visible = (
            float(visible_policy["lower_bound_height_above_cutoff_m"])
            if isinstance(visible_policy, Mapping)
            else float(gate["height_lower_bound_m"])
        )
        upper_visible = (
            float(visible_policy["upper_bound_height_above_cutoff_m"])
            if isinstance(visible_policy, Mapping)
            else float(gate["height_upper_bound_m"])
        )
        expected_visible = (
            float(visible_policy["expected_height_above_cutoff_m"])
            if isinstance(visible_policy, Mapping)
            else float(gate["expected_height_m"])
        )
        visible_passed = lower_visible <= visible_height_above_cutoff <= upper_visible
        visible_evidence = {
            "required": True,
            "passed": visible_passed,
            "status": "measured",
            **surface,
            "observed_height_m": visible_height,
            "observed_total_height_m": visible_height,
            "observed_height_above_cutoff_m": visible_height_above_cutoff,
            "opaque_base_cutoff_height_m": cutoff_height,
            "lower_bound_height_above_cutoff_m": lower_visible,
            "upper_bound_height_above_cutoff_m": upper_visible,
            "expected_height_above_cutoff_m": expected_visible,
            "absolute_error_m": abs(
                visible_height - float(gate["expected_height_m"])
            ),
            "absolute_error_above_cutoff_m": abs(
                visible_height_above_cutoff - expected_visible
            ),
        }
    if (
        not visible_passed
        and "visible_height_outside_geometry_volume_bounds" not in failures
    ):
        failures.append("visible_height_outside_geometry_volume_bounds")
    gate.update(
        {
            "visible_passed": visible_passed,
            "visible_mask_evidence": visible_evidence,
            "failure_reasons": failures,
            "passed": gate.get("state_passed") is True and visible_passed,
        }
    )
    return result


def _persistent_direct_render_product_update_gating_valid(
    receipt: Mapping[str, object]
) -> bool:
    """Validate one enabled product paired with direct 120 Hz physics stepping."""

    render = receipt.get("render")
    timing = receipt.get("timing")
    memory = receipt.get("memory")
    if not all(isinstance(row, Mapping) for row in (render, timing, memory)):
        return False
    gating = render.get("render_product_update_gating")
    phase_timings = render.get("phase_timings_s")
    if not isinstance(gating, Mapping) or not isinstance(phase_timings, Mapping):
        return False

    expected_timing_phases = {
        "warmup_app_update",
        "physics_direct_step",
        "diffusion",
        "palette_mapping",
        "prototype_authoring",
        "capture_rgb_orchestrator",
        "capture_semantic_orchestrator",
        "capture_artifact_io",
    }
    if set(phase_timings) != expected_timing_phases:
        return False
    for row in phase_timings.values():
        if not isinstance(row, Mapping):
            return False
        count = row.get("count")
        total = row.get("total")
        maximum = row.get("maximum")
        if (
            type(count) is not int
            or count <= 0
            or not _finite_number(total)
            or float(total) < 0.0
            or not _finite_number(maximum)
            or float(maximum) < 0.0
            or float(maximum) > float(total) + 1.0e-12
        ):
            return False

    physics_step_count = timing.get("callbacks_observed")
    frame_count = render.get("frame_count")
    frame_steps = render.get("frame_physics_steps")
    if (
        type(physics_step_count) is not int
        or physics_step_count <= 0
        or type(frame_count) is not int
        or frame_count <= 0
        or not isinstance(frame_steps, list)
        or len(frame_steps) != frame_count
        or any(type(step) is not int or step <= 0 for step in frame_steps)
    ):
        return False
    if any(
        phase_timings[phase].get("count") != physics_step_count
        for phase in (
            "physics_direct_step",
            "diffusion",
            "palette_mapping",
            "prototype_authoring",
        )
    ):
        return False
    if (
        phase_timings["warmup_app_update"].get("count") != 1
        or phase_timings["capture_artifact_io"].get("count") != frame_count
    ):
        return False

    passes = gating.get("explicit_render_passes")
    explicit_count = gating.get("explicit_render_pass_count")
    rotation_disproof = gating.get("rotation_disproof")
    if (
        gating.get("mode") != "persistent_enabled_direct_physics"
        or gating.get("documented_source")
        != RENDER_PRODUCT_GATING_DOCUMENTATION_URL
        or gating.get("physics_step_documented_source")
        != SIMULATION_MANAGER_STEP_DOCUMENTATION_URL
        or gating.get("initial_enabled_readback") is not True
        or gating.get("final_enabled_readback") is not True
        or gating.get("current_updates_enabled") is not True
        or gating.get("persistent_enabled") is not True
        or gating.get("rotation_enabled") is not False
        or not isinstance(rotation_disproof, Mapping)
        or rotation_disproof.get("run_id") != "C6"
        or rotation_disproof.get("status") != "failed"
        or not _valid_sha256(rotation_disproof.get("receipt_sha256"))
        or gating.get("render_product_create_count") != 1
        or gating.get("physics_steps_while_enabled") != physics_step_count
        or gating.get("physics_loop_app_update_count") != 0
        or gating.get("update_toggle_request_count") != 0
        or type(explicit_count) is not int
        or explicit_count <= 1
        or not isinstance(passes, list)
        or len(passes) != explicit_count
        or gating.get("renderer_accumulation_reset_count")
        != explicit_count - 1
        or gating.get("rotation_count") not in (None, 0)
        or gating.get("enable_request_count") not in (None, 0)
        or gating.get("disable_request_count") not in (None, 0)
        or gating.get("transitions") not in (None, [])
        or gating.get("rotations") not in (None, [])
    ):
        return False
    allowed_pass_timings = {
        "warmup_app_update",
        "capture_rgb_orchestrator",
        "capture_semantic_orchestrator",
    }
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("phase"), str)
        or not row.get("phase")
        or row.get("timing_phase") not in allowed_pass_timings
        or not _finite_number(row.get("elapsed_seconds"))
        or float(row.get("elapsed_seconds", -1.0)) < 0.0
        or row.get("enabled_before") is not True
        or row.get("enabled_after") is not True
        or row.get("passed") is not True
        or type(row.get("renderer_accumulation_reset_before")) is not bool
        for row in passes
    ):
        return False
    if (
        passes[0].get("phase") != "warmup"
        or passes[0].get("timing_phase") != "warmup_app_update"
        or passes[0].get("renderer_accumulation_reset_before") is not False
        or any(
            row.get("renderer_accumulation_reset_before") is not True
            for row in passes[1:]
        )
    ):
        return False
    pass_counts = {
        phase: sum(row.get("timing_phase") == phase for row in passes)
        for phase in allowed_pass_timings
    }
    if (
        pass_counts["warmup_app_update"] != 1
        or pass_counts["capture_rgb_orchestrator"] < frame_count + 2
        or pass_counts["capture_semantic_orchestrator"] < frame_count
        or any(
            phase_timings[phase].get("count") != pass_counts[phase]
            for phase in allowed_pass_timings
        )
    ):
        return False

    physics_hz = timing.get("physics_hz")
    callback_indices = timing.get("callback_indices")
    direct_indices = timing.get("callback_direct_step_indices")
    callback_dts = timing.get("callback_dts_s")
    callback_times = timing.get("callback_times_s")
    logical_times = timing.get("callback_logical_times_s")
    direct_initialization = timing.get("direct_step_initialization")
    multitick = timing.get("multitick_support")
    if (
        physics_hz != 120
        or timing.get("simulation_owner") != "SimulationManager.step"
        or timing.get("physics_step_method")
        != "SimulationManager.step(steps=1, update_fabric=False)"
        or timing.get("physics_step_documented_source")
        != SIMULATION_MANAGER_STEP_DOCUMENTATION_URL
        or timing.get("executed_physics_steps") != physics_step_count
        or timing.get("callbacks_expected") != physics_step_count
        or timing.get("one_callback_per_direct_step") is not True
        or timing.get("physics_loop_app_update_count") != 0
        or timing.get("direct_physx_api_used") is not True
        or timing.get("callback_contract_failure") is not None
        or not isinstance(multitick, Mapping)
        or multitick.get("enabled_readback") is not True
        or not isinstance(direct_initialization, Mapping)
        or direct_initialization.get("timeline_playing") is not False
        or direct_initialization.get("update_fabric") is not False
        or type(direct_initialization.get("baseline_num_physics_steps"))
        is not int
        or not _finite_number(
            direct_initialization.get("baseline_simulation_time_s")
        )
        or not _finite_number(
            direct_initialization.get("paused_timeline_time_s")
        )
    ):
        return False
    expected_indices = list(range(1, physics_step_count + 1))
    if callback_indices != expected_indices or direct_indices != expected_indices:
        return False
    if not all(
        isinstance(values, list) and len(values) == physics_step_count
        for values in (callback_dts, callback_times, logical_times)
    ):
        return False
    paused_time = float(direct_initialization["paused_timeline_time_s"])
    expected_dt = 1.0 / 120.0
    if any(
        not _finite_number(value)
        or not math.isclose(
            float(value), expected_dt, rel_tol=0.0, abs_tol=1.0e-6
        )
        for value in callback_dts
    ) or any(
        not _finite_number(value)
        or not math.isclose(
            float(value), paused_time, rel_tol=0.0, abs_tol=1.0e-6
        )
        for value in callback_times
    ) or any(
        not _finite_number(value)
        or not math.isclose(
            float(value), index / 120.0, rel_tol=0.0, abs_tol=1.0e-6
        )
        for index, value in enumerate(logical_times, start=1)
    ):
        return False

    emission_boundaries = timing.get("emission_boundaries")
    if (
        timing.get("emissions_authored_on_main_thread") is not True
        or timing.get("emission_boundary_verified") is not True
        or not isinstance(emission_boundaries, list)
        or not emission_boundaries
        or any(
            not isinstance(row, Mapping)
            or type(row.get("physics_step")) is not int
            or row.get("physics_step") <= 0
            or row.get("authored_on_main_thread") is not True
            or row.get("authored_after_callback_count")
            != row.get("physics_step") - 1
            or row.get("scheduled_before_callback_index")
            != row.get("physics_step")
            for row in emission_boundaries
        )
    ):
        return False

    probe = timing.get("direct_step_probe")
    if not isinstance(probe, Mapping):
        return False
    probe_rows = probe.get("step_evidence")
    if (
        probe.get("step_limit") != 12
        or probe.get("first_capture_step") != frame_steps[0]
        or probe.get("first_capture_step") < probe.get("step_limit")
        or probe.get("completed_steps") != 12
        or type(probe.get("emission_events_observed")) is not int
        or probe.get("emission_events_observed") <= 0
        or probe.get("physics_contract_passed") is not True
        or probe.get("capture_contract_passed") is not True
        or probe.get("status") != "pass"
        or not isinstance(probe_rows, list)
        or len(probe_rows) != 12
    ):
        return False
    for step, row in enumerate(probe_rows, start=1):
        if not isinstance(row, Mapping):
            return False
        particle_state = row.get("particle_state")
        particle_count = (
            particle_state.get("particle_count")
            if isinstance(particle_state, Mapping)
            else None
        )
        if (
            row.get("direct_step_index") != step
            or row.get("callback_count_delta") != 1
            or row.get("physics_step_count_delta") != 1
            or not _finite_number(row.get("simulation_time_delta_s"))
            or not math.isclose(
                float(row.get("simulation_time_delta_s", math.nan)),
                expected_dt,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            or not _finite_number(row.get("timeline_time_before_s"))
            or not math.isclose(
                float(row.get("timeline_time_before_s", math.nan)),
                paused_time,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _finite_number(row.get("timeline_time_after_s"))
            or not math.isclose(
                float(row.get("timeline_time_after_s", math.nan)),
                paused_time,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or row.get("timeline_playing_before") is not False
            or row.get("timeline_playing_after") is not False
            or not isinstance(row.get("callback_contract"), Mapping)
            or row["callback_contract"].get("passed") is not True
            or row.get("failed_predicates") != []
            or row.get("passed") is not True
            or not isinstance(particle_state, Mapping)
            or type(particle_count) is not int
            or particle_count < 0
            or particle_state.get("positions_shape") != [particle_count, 3]
            or particle_state.get("velocities_shape") != [particle_count, 3]
            or particle_state.get("finite_positions") is not True
            or particle_state.get("finite_velocities") is not True
            or particle_state.get("ids_exact") is not True
            or particle_state.get("passed") is not True
        ):
            return False

    capture_guards = timing.get("capture_guard_records")
    if (
        timing.get("capture_callbacks_advanced") != 0
        or timing.get("capture_direct_physics_steps_advanced") != 0
        or timing.get("capture_simulation_time_advanced") is not False
        or timing.get("capture_timeline_time_advanced") is not False
        or timing.get("capture_playback_preserved") is not True
        or not isinstance(capture_guards, list)
        or len(capture_guards) != explicit_count - 1
    ):
        return False
    if any(
        not isinstance(row, Mapping)
        or type(row.get("callbacks_before")) is not int
        or row.get("callbacks_after") != row.get("callbacks_before")
        or type(row.get("physics_steps_before")) is not int
        or row.get("physics_steps_after") != row.get("physics_steps_before")
        or not _finite_number(row.get("simulation_time_before_s"))
        or not _finite_number(row.get("simulation_time_after_s"))
        or not math.isclose(
            float(row.get("simulation_time_before_s", math.nan)),
            float(row.get("simulation_time_after_s", math.nan)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not _finite_number(row.get("timeline_time_before_s"))
        or not _finite_number(row.get("timeline_time_after_s"))
        or not math.isclose(
            float(row.get("timeline_time_before_s", math.nan)),
            float(row.get("timeline_time_after_s", math.nan)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or type(row.get("timeline_playing_before")) is not bool
        or row.get("timeline_playing_after")
        is not row.get("timeline_playing_before")
        for row in capture_guards
    ):
        return False

    interval = memory.get("periodic_capture_interval_frames")
    periodic_samples = memory.get("periodic_capture_samples")
    all_memory_samples = memory.get("samples_mib")
    expected_completed_counts = list(
        range(
            PERIODIC_CAPTURE_MEMORY_INTERVAL_FRAMES,
            frame_count + 1,
            PERIODIC_CAPTURE_MEMORY_INTERVAL_FRAMES,
        )
    )
    if (
        memory.get("hard_limit_mib") != GPU_MEMORY_HARD_LIMIT_MIB
        or memory.get("hard_limit_exceeded") is not False
        or interval != PERIODIC_CAPTURE_MEMORY_INTERVAL_FRAMES
        or not isinstance(periodic_samples, list)
        or len(periodic_samples) != len(expected_completed_counts)
        or not isinstance(all_memory_samples, list)
    ):
        return False
    for completed_count, row in zip(expected_completed_counts, periodic_samples):
        if not isinstance(row, Mapping):
            return False
        used_memory = row.get("used_gpu_memory_mib")
        if (
            row.get("completed_frame_count") != completed_count
            or row.get("physics_step") != frame_steps[completed_count - 1]
            or type(used_memory) is not int
            or used_memory < 0
            or used_memory >= GPU_MEMORY_HARD_LIMIT_MIB
            or used_memory not in all_memory_samples
        ):
            return False
    return True


def _legacy_render_product_update_gating_valid(
    receipt: Mapping[str, object]
) -> bool:
    """Validate documented physics-only render-product gating and timing evidence."""

    render = receipt.get("render")
    timing = receipt.get("timing")
    if not isinstance(render, Mapping) or not isinstance(timing, Mapping):
        return False
    gating = render.get("render_product_update_gating")
    phase_timings = render.get("phase_timings_s")
    if not isinstance(gating, Mapping) or not isinstance(phase_timings, Mapping):
        return False
    expected_timing_phases = {
        "warmup_app_update",
        "physics_app_update",
        "diffusion",
        "palette_mapping",
        "prototype_authoring",
        "capture_rgb_orchestrator",
        "capture_semantic_orchestrator",
        "capture_artifact_io",
    }
    if set(phase_timings) != expected_timing_phases:
        return False
    for row in phase_timings.values():
        if not isinstance(row, Mapping):
            return False
        count = row.get("count")
        total = row.get("total")
        maximum = row.get("maximum")
        if (
            type(count) is not int
            or count <= 0
            or not _finite_number(total)
            or float(total) < 0.0
            or not _finite_number(maximum)
            or float(maximum) < 0.0
            or float(maximum) > float(total) + 1.0e-12
        ):
            return False

    physics_step_count = timing.get("callbacks_observed")
    frame_count = render.get("frame_count")
    if (
        type(physics_step_count) is not int
        or physics_step_count <= 0
        or type(frame_count) is not int
        or frame_count <= 0
    ):
        return False
    if any(
        phase_timings[phase].get("count") != physics_step_count
        for phase in (
            "physics_app_update",
            "diffusion",
            "palette_mapping",
            "prototype_authoring",
        )
    ):
        return False
    if (
        phase_timings["warmup_app_update"].get("count") != 1
        or phase_timings["capture_artifact_io"].get("count") != frame_count
    ):
        return False

    transitions = gating.get("transitions")
    passes = gating.get("explicit_render_passes")
    explicit_count = gating.get("explicit_render_pass_count")
    enable_count = gating.get("enable_request_count")
    disable_count = gating.get("disable_request_count")
    rotations = gating.get("rotations")
    rotation_count = gating.get("rotation_count")
    expected_rotation_count = (
        (frame_count - 1) // RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES
    )
    if (
        gating.get("documented_source")
        != RENDER_PRODUCT_GATING_DOCUMENTATION_URL
        or gating.get("cleanup_documented_source")
        != RENDER_PRODUCT_CLEANUP_DOCUMENTATION_URL
        or gating.get("initial_disabled_readback") is not True
        or gating.get("final_disabled_readback") is not True
        or gating.get("current_updates_enabled") is not False
        or gating.get("enabled_only_for_explicit_render_passes") is not True
        or gating.get("rotation_interval_logical_frames")
        != RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES
        or rotation_count != expected_rotation_count
        or not isinstance(rotations, list)
        or len(rotations) != expected_rotation_count
        or gating.get("physics_updates_while_disabled") != physics_step_count
        or type(explicit_count) is not int
        or explicit_count <= 0
        or enable_count != explicit_count
        or disable_count != explicit_count + 2 + expected_rotation_count
        or not isinstance(transitions, list)
        or len(transitions) != enable_count + disable_count
        or not isinstance(passes, list)
        or len(passes) != explicit_count
    ):
        return False
    for index, row in enumerate(rotations, start=1):
        if not isinstance(row, Mapping):
            return False
        memory_before = row.get("memory_before_destroy_mib")
        memory_after = row.get("memory_after_destroy_mib")
        memory_released = row.get("memory_released_mib")
        flush = row.get("flush_invariants")
        if (
            row.get("completed_frame_count")
            != index * RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES
            or row.get("generation_before") != index - 1
            or row.get("generation_after") != index
            or row.get("wait_until_complete") is not True
            or row.get("annotator_count") != 3
            or row.get("detached_annotator_count") != 3
            or row.get("destroy_called") is not True
            or not isinstance(flush, Mapping)
            or flush.get("callbacks_unchanged") is not True
            or flush.get("timeline_time_unchanged") is not True
            or flush.get("playback_restored") is not True
            or flush.get("passed") is not True
            or type(memory_before) is not int
            or type(memory_after) is not int
            or type(memory_released) is not int
            or memory_before < 0
            or memory_after < 0
            or memory_released != memory_before - memory_after
            or memory_released <= 0
            or row.get("measurable_memory_release") is not True
            or row.get("new_product_disabled_readback") is not True
            or row.get("reattached_annotator_count") != 3
            or row.get("passed") is not True
        ):
            return False
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("phase"), str)
        or not row.get("phase")
        or type(row.get("requested_updates_enabled")) is not bool
        or type(row.get("updates_enabled_before")) is not bool
        or type(row.get("updates_enabled_after")) is not bool
        or row.get("readback_matches_request") is not True
        or row.get("updates_enabled_after")
        is not row.get("requested_updates_enabled")
        or not _finite_number(row.get("set_and_readback_seconds"))
        or float(row.get("set_and_readback_seconds", -1.0)) < 0.0
        for row in transitions
    ):
        return False
    expected_rotation_phases = [
        f"rotation_generation_{index:02d}_after_create"
        for index in range(1, expected_rotation_count + 1)
    ]
    rotation_phase_set = set(expected_rotation_phases)
    rotation_transitions = [
        row
        for row in transitions
        if row.get("phase") in rotation_phase_set
    ]
    if (
        sum(row["requested_updates_enabled"] is True for row in transitions)
        != enable_count
        or sum(row["requested_updates_enabled"] is False for row in transitions)
        != disable_count
        or transitions[0].get("phase") != "initial_after_create"
        or transitions[0].get("requested_updates_enabled") is not False
        or transitions[-1].get("phase") != "final_before_acceptance"
        or transitions[-1].get("requested_updates_enabled") is not False
        or [row.get("phase") for row in rotation_transitions]
        != expected_rotation_phases
        or any(
            row.get("requested_updates_enabled") is not False
            or row.get("updates_enabled_after") is not False
            for row in rotation_transitions
        )
    ):
        return False
    core_transitions = [
        row for row in transitions if row.get("phase") not in rotation_phase_set
    ]
    if any(
        current.get("updates_enabled_before")
        is not previous.get("updates_enabled_after")
        for previous, current in zip(core_transitions, core_transitions[1:])
    ):
        return False
    for pass_index in range(explicit_count):
        enable_transition = core_transitions[1 + 2 * pass_index]
        disable_transition = core_transitions[2 + 2 * pass_index]
        if (
            enable_transition.get("requested_updates_enabled") is not True
            or enable_transition.get("updates_enabled_before") is not False
            or disable_transition.get("requested_updates_enabled") is not False
            or disable_transition.get("updates_enabled_before") is not True
        ):
            return False

    allowed_pass_timings = {
        "warmup_app_update",
        "capture_rgb_orchestrator",
        "capture_semantic_orchestrator",
    }
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("phase"), str)
        or not row.get("phase")
        or row.get("timing_phase") not in allowed_pass_timings
        or not _finite_number(row.get("elapsed_seconds"))
        or float(row.get("elapsed_seconds", -1.0)) < 0.0
        or row.get("passed") is not True
        or row.get("disabled_after") is not True
        for row in passes
    ):
        return False
    pass_counts = {
        phase: sum(row.get("timing_phase") == phase for row in passes)
        for phase in allowed_pass_timings
    }
    if (
        passes[0].get("phase") != "warmup"
        or pass_counts["warmup_app_update"] != 1
        or pass_counts["capture_rgb_orchestrator"] < frame_count + 2
        or pass_counts["capture_semantic_orchestrator"] < frame_count
        or any(
            phase_timings[phase].get("count") != pass_counts[phase]
            for phase in allowed_pass_timings
        )
    ):
        return False
    return True


def _timeline_substep_timing_contract_valid(
    timing: Mapping[str, object], *, expected_report_interval_count: int
) -> bool:
    """Validate the accepted two-substep report/PhysX clock evidence."""

    if (
        not isinstance(timing, Mapping)
        or type(expected_report_interval_count) is not int
        or expected_report_interval_count <= 0
    ):
        return False
    report_hz = timing.get("report_hz")
    substeps = timing.get("physics_substeps_per_report_interval")
    if report_hz != 120 or substeps != 2:
        return False
    physics_hz = report_hz * substeps
    callback_count = expected_report_interval_count * substeps
    expected_duration_s = expected_report_interval_count / float(report_hz)
    callback_indices = timing.get("callback_indices")
    app_update_indices = timing.get("callback_app_update_indices")
    callback_dts = timing.get("callback_dts_s")
    callback_times = timing.get("callback_times_s")
    logical_times = timing.get("callback_logical_times_s")
    expected_callback_indices = list(range(1, callback_count + 1))
    expected_app_update_indices = [
        report_index
        for report_index in range(1, expected_report_interval_count + 1)
        for _ in range(substeps)
    ]
    if (
        not _finite_number(timing.get("report_dt_s"))
        or not math.isclose(
            float(timing["report_dt_s"]), 1.0 / report_hz, rel_tol=0.0, abs_tol=1e-12
        )
        or timing.get("report_interval_count") != expected_report_interval_count
        or timing.get("executed_report_intervals") != expected_report_interval_count
        or timing.get("physics_hz") != physics_hz
        or timing.get("physics_steps_per_second") != physics_hz
        or timing.get("physics_callback_count") != callback_count
        or not _finite_number(timing.get("physics_dt_s"))
        or not math.isclose(
            float(timing["physics_dt_s"]), 1.0 / physics_hz, rel_tol=0.0, abs_tol=1e-12
        )
        or timing.get("min_frame_rate") != report_hz
        or timing.get("timeline_ticks_per_frame") != 1
        or not _finite_number(timing.get("timeline_end_s"))
        or not math.isclose(
            float(timing["timeline_end_s"]),
            (expected_report_interval_count + 1) / float(report_hz),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not _finite_number(timing.get("physical_duration_s"))
        or not math.isclose(
            float(timing["physical_duration_s"]),
            expected_duration_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or timing.get("executed_physics_steps") != callback_count
        or timing.get("callbacks_expected") != callback_count
        or timing.get("callbacks_observed") != callback_count
        or callback_indices != expected_callback_indices
        or app_update_indices != expected_app_update_indices
        or timing.get("one_callback_per_app_update") is not False
        or timing.get("callbacks_per_app_update_contract_passed") is not True
        or timing.get("simulation_owner") != "timeline_app_update"
        or timing.get("physics_step_method") != "app.update()"
        or timing.get("direct_physx_api_used") is not False
        or timing.get("physics_loop_app_update_count")
        != expected_report_interval_count
        or timing.get("callback_group_trace_count")
        != expected_report_interval_count
        or not isinstance(timing.get("callback_group_trace_path"), str)
        or not timing.get("callback_group_trace_path")
        or not _valid_sha256(timing.get("callback_group_trace_sha256"))
        or timing.get("final_edit_target_is_session") is not True
        or any(
            timing.get(key) is not True
            for key in (
                "positions_session_layer_only",
                "velocities_session_layer_only",
                "orientations_session_layer_only",
                "scales_session_layer_only",
                "proto_indices_session_layer_only",
                "ids_session_layer_only",
            )
        )
        or timing.get("simulation_points_root_layer_absent") is not True
        or timing.get("simulation_points_runtime_readback_passed") is not True
    ):
        return False
    if not all(
        isinstance(values, list) and len(values) == callback_count
        for values in (callback_dts, callback_times, logical_times)
    ):
        return False
    if any(
        not _finite_number(value)
        or not math.isclose(
            float(value), 1.0 / physics_hz, rel_tol=0.0, abs_tol=1e-6
        )
        for value in callback_dts
    ):
        return False
    if any(not _finite_number(value) for value in callback_times) or any(
        float(current) < float(previous) - 1e-12
        for previous, current in zip(callback_times, callback_times[1:])
    ):
        return False
    if any(
        not _finite_number(value)
        or not math.isclose(
            float(value), index / float(physics_hz), rel_tol=0.0, abs_tol=1e-9
        )
        for index, value in enumerate(logical_times, start=1)
    ):
        return False
    if (
        not _finite_number(timing.get("callback_dt_sum_s"))
        or not math.isclose(
            float(timing["callback_dt_sum_s"]),
            expected_duration_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or timing.get("diffusion_updates_per_report_interval") != 1
        or timing.get("diffusion_update_count") != expected_report_interval_count
        or not _finite_number(timing.get("diffusion_dt_s"))
        or not math.isclose(
            float(timing["diffusion_dt_s"]),
            1.0 / report_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not _finite_number(timing.get("diffusion_integrated_time_s"))
        or not math.isclose(
            float(timing["diffusion_integrated_time_s"]),
            expected_duration_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        return False

    manager = timing.get("physics_manager_readback")
    if not isinstance(manager, Mapping):
        return False
    manager_count_fields = (
        manager.get("baseline_num_physics_steps"),
        manager.get("final_num_physics_steps"),
        manager.get("delta_num_physics_steps"),
    )
    manager_time_fields = (
        manager.get("baseline_simulation_time_s"),
        manager.get("final_simulation_time_s"),
        manager.get("delta_simulation_time_s"),
    )
    if (
        any(type(value) is not int for value in manager_count_fields)
        or manager_count_fields[1] - manager_count_fields[0] != callback_count
        or manager_count_fields[2] != callback_count
        or any(not _finite_number(value) for value in manager_time_fields)
        or not math.isclose(
            float(manager_time_fields[1]) - float(manager_time_fields[0]),
            expected_duration_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or not math.isclose(
            float(manager_time_fields[2]),
            expected_duration_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
        or manager.get("passed") is not True
    ):
        return False

    settings = timing.get("physics_settings_readback")
    if (
        not isinstance(settings, Mapping)
        or not isinstance(settings.get("physics_scene_path"), str)
        or not settings.get("physics_scene_path")
        or settings.get("requested_time_steps_per_second") != physics_hz
        or settings.get("readback_time_steps_per_second") != physics_hz
        or settings.get("requested_min_frame_rate") != report_hz
        or settings.get("readback_min_frame_rate") != report_hz
        or settings.get("requested_fixed_time_stepping") is not True
        or settings.get("readback_fixed_time_stepping") is not True
        or settings.get("requested_timeline_rate") != report_hz
        or not _finite_number(settings.get("readback_timeline_rate"))
        or not math.isclose(
            float(settings["readback_timeline_rate"]),
            float(report_hz),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or settings.get("requested_timeline_ticks_per_frame") != 1
        or settings.get("readback_timeline_ticks_per_frame") != 1
        or settings.get("passed") is not True
    ):
        return False

    boundaries = timing.get("emission_boundaries")
    if (
        timing.get("emissions_authored_on_main_thread") is not True
        or timing.get("emission_boundary_verified") is not True
        or not isinstance(boundaries, list)
        or not boundaries
    ):
        return False
    for row in boundaries:
        if not isinstance(row, Mapping):
            return False
        report_step = row.get("report_step")
        if type(report_step) is not int or report_step <= 0:
            return False
        first_callback = (report_step - 1) * substeps + 1
        if (
            row.get("physics_step") != report_step
            or row.get("authored_after_callback_count") != first_callback - 1
            or row.get("scheduled_before_callback_index") != first_callback
            or row.get("scheduled_before_callback_indices")
            != list(range(first_callback, first_callback + substeps))
            or row.get("authored_on_main_thread") is not True
        ):
            return False
    return True


def _timeline_app_update_render_product_gating_valid(
    receipt: Mapping[str, object],
) -> bool:
    """Validate timeline-owned physics with one gated render product."""

    render = receipt.get("render")
    timing = receipt.get("timing")
    if not isinstance(render, Mapping) or not isinstance(timing, Mapping):
        return False
    gating = render.get("render_product_update_gating")
    phase_timings = render.get("phase_timings_s")
    if not isinstance(gating, Mapping) or not isinstance(phase_timings, Mapping):
        return False
    expected_timing_phases = {
        "warmup_app_update",
        "physics_app_update",
        "diffusion",
        "palette_mapping",
        "prototype_authoring",
        "capture_rgb_orchestrator",
        "capture_semantic_orchestrator",
        "capture_artifact_io",
    }
    if set(phase_timings) != expected_timing_phases:
        return False
    if any(
        not isinstance(row, Mapping)
        or type(row.get("count")) is not int
        or int(row.get("count", 0)) <= 0
        or not _finite_number(row.get("total"))
        or float(row.get("total", -1.0)) < 0.0
        or not _finite_number(row.get("maximum"))
        or float(row.get("maximum", -1.0)) < 0.0
        or float(row.get("maximum", math.inf))
        > float(row.get("total", -math.inf)) + 1.0e-12
        for row in phase_timings.values()
    ):
        return False

    callback_count = timing.get("callbacks_observed")
    frame_count = render.get("frame_count")
    physics_substeps = timing.get("physics_substeps_per_report_interval", 1)
    report_interval_count = timing.get(
        "report_interval_count",
        callback_count if physics_substeps == 1 else None,
    )
    if (
        type(callback_count) is not int
        or callback_count <= 0
        or type(physics_substeps) is not int
        or physics_substeps not in {1, 2}
        or type(report_interval_count) is not int
        or report_interval_count <= 0
        or callback_count != report_interval_count * physics_substeps
        or type(frame_count) is not int
        or frame_count <= 0
        or any(
            phase_timings[phase].get("count") != report_interval_count
            for phase in (
                "physics_app_update",
                "diffusion",
                "palette_mapping",
                "prototype_authoring",
            )
        )
        or phase_timings["warmup_app_update"].get("count") != 1
        or phase_timings["capture_artifact_io"].get("count") != frame_count
    ):
        return False

    explicit_count = gating.get("explicit_render_pass_count")
    enable_count = gating.get("enable_request_count")
    disable_count = gating.get("disable_request_count")
    transitions = gating.get("transitions")
    passes = gating.get("explicit_render_passes")
    rotation_disproof = gating.get("rotation_disproof")
    if (
        gating.get("mode") != "timeline_app_update_explicit_capture"
        or gating.get("documented_source")
        != RENDER_PRODUCT_GATING_DOCUMENTATION_URL
        or gating.get("physics_step_documented_source")
        != TIMELINE_APP_UPDATE_DOCUMENTATION_URL
        or gating.get("initial_disabled_readback") is not True
        or gating.get("final_disabled_readback") is not True
        or gating.get("current_updates_enabled") is not False
        or gating.get("enabled_only_for_explicit_render_passes") is not True
        or gating.get("rotation_enabled") is not False
        or not isinstance(rotation_disproof, Mapping)
        or rotation_disproof.get("run_id") != "C6"
        or rotation_disproof.get("status") != "failed"
        or not _valid_sha256(rotation_disproof.get("receipt_sha256"))
        or gating.get("render_product_create_count") != 1
        or gating.get("physics_updates_while_disabled") != report_interval_count
        or gating.get("physics_loop_app_update_count") != report_interval_count
        or type(explicit_count) is not int
        or explicit_count <= 1
        or enable_count != explicit_count
        or disable_count != explicit_count + 2
        or not isinstance(transitions, list)
        or len(transitions) != enable_count + disable_count
        or not isinstance(passes, list)
        or len(passes) != explicit_count
        or gating.get("renderer_accumulation_reset_count")
        != explicit_count - 1
    ):
        return False
    if any(
        not isinstance(row, Mapping)
        or type(row.get("requested_updates_enabled")) is not bool
        or row.get("updates_enabled_after")
        is not row.get("requested_updates_enabled")
        or row.get("readback_matches_request") is not True
        or not _finite_number(row.get("set_and_readback_seconds"))
        or float(row.get("set_and_readback_seconds", -1.0)) < 0.0
        for row in transitions
    ):
        return False
    if (
        transitions[0].get("phase") != "initial_after_create"
        or transitions[0].get("requested_updates_enabled") is not False
        or transitions[-1].get("phase") != "final_before_acceptance"
        or transitions[-1].get("requested_updates_enabled") is not False
        or sum(
            row.get("requested_updates_enabled") is True
            for row in transitions
        )
        != enable_count
        or sum(
            row.get("requested_updates_enabled") is False
            for row in transitions
        )
        != disable_count
    ):
        return False
    if any(
        not isinstance(row, Mapping)
        or row.get("passed") is not True
        or row.get("disabled_after") is not True
        or row.get("timing_phase")
        not in {
            "warmup_app_update",
            "capture_rgb_orchestrator",
            "capture_semantic_orchestrator",
        }
        or not _finite_number(row.get("elapsed_seconds"))
        for row in passes
    ):
        return False

    callback_indices = timing.get("callback_indices")
    app_update_indices = timing.get("callback_app_update_indices")
    capture_guards = timing.get("capture_guard_records")
    motion = timing.get("motion_falsifier")
    frame_steps = render.get("frame_physics_steps")
    expected_indices = list(range(1, callback_count + 1))
    expected_app_update_indices = [
        report_index
        for report_index in range(1, report_interval_count + 1)
        for _ in range(physics_substeps)
    ]
    callback_owner_valid = bool(
        (
            physics_substeps == 1
            and timing.get("one_callback_per_app_update") is True
        )
        or (
            physics_substeps == 2
            and _timeline_substep_timing_contract_valid(
                timing,
                expected_report_interval_count=report_interval_count,
            )
        )
    )
    if (
        timing.get("simulation_owner") != "timeline_app_update"
        or timing.get("physics_step_method") != "app.update()"
        or timing.get("physics_step_documented_source")
        != TIMELINE_APP_UPDATE_DOCUMENTATION_URL
        or not callback_owner_valid
        or timing.get("physics_loop_app_update_count") != report_interval_count
        or timing.get("direct_physx_api_used") is not False
        or callback_indices != expected_indices
        or app_update_indices != expected_app_update_indices
        or timing.get("capture_callbacks_advanced") != 0
        or timing.get("capture_timeline_time_advanced") is not False
        or timing.get("capture_playback_preserved") is not True
        or not isinstance(capture_guards, list)
        or len(capture_guards) != explicit_count - 1
        or not isinstance(motion, Mapping)
        or motion.get("probe_step") != F2_MOTION_FALSIFIER_STEP
        or motion.get("status") != "pass"
        or motion.get("passed") is not True
        or not _valid_sha256(motion.get("artifact_sha256"))
        or not isinstance(motion.get("artifact_path"), str)
        or not motion.get("artifact_path")
        or not isinstance(frame_steps, list)
        or len(frame_steps) != frame_count
        or any(type(step) is not int or step <= 0 for step in frame_steps)
        or frame_steps[0] < motion.get("probe_step")
    ):
        return False
    return all(
        isinstance(row, Mapping)
        and row.get("callbacks_before") == row.get("callbacks_after")
        and math.isclose(
            float(row.get("timeline_time_before_s", math.nan)),
            float(row.get("timeline_time_after_s", math.nan)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and row.get("timeline_playing_before")
        is row.get("timeline_playing_after")
        for row in capture_guards
    )


def _render_product_update_gating_valid(receipt: Mapping[str, object]) -> bool:
    """Validate the current direct path or retained historical receipts."""

    render = receipt.get("render")
    gating = render.get("render_product_update_gating") if isinstance(render, Mapping) else None
    if (
        isinstance(gating, Mapping)
        and gating.get("mode") == "persistent_enabled_direct_physics"
    ):
        return _persistent_direct_render_product_update_gating_valid(receipt)
    if (
        isinstance(gating, Mapping)
        and gating.get("mode")
        == "timeline_app_update_explicit_capture"
    ):
        return _timeline_app_update_render_product_gating_valid(receipt)
    return _legacy_render_product_update_gating_valid(receipt)


def _runtime_particle_layer_contract_valid(receipt: Mapping[str, object]) -> bool:
    """Validate authored runtime rows separately from read-only PhysX state."""

    instancer = receipt.get("point_instancer")
    timing = receipt.get("timing")
    plan = receipt.get("plan")
    captures = receipt.get("captures")
    if (
        not isinstance(instancer, Mapping)
        or not isinstance(timing, Mapping)
        or not isinstance(plan, Mapping)
        or not isinstance(captures, list)
        or not captures
    ):
        return False
    layer = instancer.get("runtime_layer_evidence")
    if not isinstance(layer, Mapping):
        return False
    specs = layer.get("property_specs")
    simulation = layer.get("simulation_points_runtime_evidence")
    totals = plan.get("totals")
    final_capture = captures[-1]
    if (
        not isinstance(specs, Mapping)
        or not isinstance(simulation, Mapping)
        or not isinstance(totals, Mapping)
        or not isinstance(final_capture, Mapping)
    ):
        return False
    expected_count = totals.get("particle_count")
    contained_count = final_capture.get("inside_vial_interior_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 0
        or isinstance(contained_count, bool)
        or not isinstance(contained_count, int)
        or contained_count < 0
    ):
        return False
    authored_properties = (
        "positions",
        "velocities",
        "orientations",
        "scales",
        "protoIndices",
        "ids",
    )
    authored_specs_valid = all(
        specs.get(f"{name}_session_layer_spec") is True
        and specs.get(f"{name}_root_layer_spec") is False
        for name in authored_properties
    )
    timing_authored_valid = all(
        timing.get(f"{name}_session_layer_only") is True
        for name in (
            "positions",
            "velocities",
            "orientations",
            "scales",
            "proto_indices",
            "ids",
        )
    )
    return bool(
        authored_specs_valid
        and timing_authored_valid
        and isinstance(specs.get("simulationPoints_session_layer_spec"), bool)
        and specs.get("simulationPoints_root_layer_spec") is False
        and layer.get("authored_session_only_properties")
        == list(authored_properties)
        and layer.get("read_only_runtime_property") == "simulationPoints"
        and layer.get("read_only_runtime_property_session_spec_required") is False
        and layer.get("read_only_runtime_property_root_spec_forbidden") is True
        and layer.get("read_only_runtime_property_documentation_url")
        == SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL
        and layer.get("failed_properties") == []
        and layer.get("final_edit_target_is_session") is True
        and layer.get("passed") is True
        and simulation.get("attribute") == "physxParticle:simulationPoints"
        and simulation.get("documented_read_only") is True
        and simulation.get("documentation_url")
        == SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL
        and simulation.get("runtime_readback_required") is True
        and simulation.get("runtime_readback_shape") == [expected_count, 3]
        and simulation.get("runtime_readback_row_count") == expected_count
        and simulation.get("expected_particle_count") == expected_count
        and simulation.get("particle_id_count") == expected_count
        and simulation.get("containment_particle_count") == contained_count
        and contained_count == expected_count
        and simulation.get("finite_xyz_rows") is True
        and simulation.get("row_counts_match") is True
        and simulation.get("passed") is True
        and timing.get("final_edit_target_is_session") is True
        and timing.get("simulation_points_root_layer_absent") is True
        and timing.get("simulation_points_runtime_readback_passed") is True
    )


def validate_paint_ball_receipt(receipt: Mapping[str, object]) -> tuple[str, ...]:
    if not isinstance(receipt, Mapping):
        return ("receipt_missing_or_unreadable",)
    failures: list[str] = []
    if receipt.get("schema_version") != PAINT_BALL_RECEIPT_SCHEMA_VERSION:
        failures.append("receipt_schema")
    if receipt.get("experiment") != "physx_three_pour_f2_paint_ball_emitter":
        failures.append("receipt_identity")
    render_path = receipt.get("render_path")
    count_driven_body = render_path == COUNT_DRIVEN_RENDER_PATH
    if render_path not in {EMITTER_RENDER_PATH, COUNT_DRIVEN_RENDER_PATH}:
        failures.append("render_path")
    if receipt.get("status") not in {"pass", "insufficient_evidence"}:
        failures.append("run_status")
    if not _render_product_update_gating_valid(receipt):
        failures.append("render_product_update_gating")
    system = receipt.get("particle_system")
    runtime_configuration = (
        system.get("runtime_configuration") if isinstance(system, Mapping) else None
    )
    arguments = receipt.get("arguments")
    arguments = arguments if isinstance(arguments, Mapping) else {}
    plan = receipt.get("plan")
    plan = plan if isinstance(plan, Mapping) else {}
    plan_totals = plan.get("totals")
    plan_totals = plan_totals if isinstance(plan_totals, Mapping) else {}
    declared_solver_iterations = arguments.get("solver_position_iterations", 4)
    if (
        isinstance(declared_solver_iterations, bool)
        or not isinstance(declared_solver_iterations, int)
        or declared_solver_iterations <= 0
    ):
        declared_solver_iterations = -1
    declared_physics_substeps = arguments.get("physics_substeps", 1)
    if (
        type(declared_physics_substeps) is not int
        or declared_physics_substeps not in {1, 2}
    ):
        declared_physics_substeps = -1
    declared_physics_steps_per_second = 120 * declared_physics_substeps
    declared_resolution_profile_id = arguments.get(
        "particle_resolution_profile", "c20_2p5mm"
    )
    try:
        declared_resolution_profile = particle_resolution_profile(
            declared_resolution_profile_id
        )
    except (TypeError, ValueError):
        declared_resolution_profile = None
    if declared_resolution_profile_id == C21_RESOLUTION_PROFILE_ID:
        saved_profile = receipt.get("particle_resolution_profile")
        expected_profile = {
            **particle_resolution_profile(C21_RESOLUTION_PROFILE_ID),
            "source_sampler": "resolution_aware_c20_boundary",
        }
        if saved_profile != expected_profile:
            failures.append("particle_resolution_profile")
        source_geometry = receipt.get("source_emission_geometry")
        source_events = (
            source_geometry.get("events")
            if isinstance(source_geometry, Mapping)
            else None
        )
        if (
            not isinstance(source_geometry, Mapping)
            or source_geometry.get("profile_id") != C21_RESOLUTION_PROFILE_ID
            or isinstance(source_geometry.get("expected_event_count"), bool)
            or not isinstance(source_geometry.get("expected_event_count"), int)
            or source_geometry.get("expected_event_count", 0) <= 0
            or source_geometry.get("observed_event_count")
            != source_geometry.get("expected_event_count")
            or not isinstance(source_events, list)
            or len(source_events) != source_geometry.get("expected_event_count")
            or source_geometry.get("all_support_gates_passed") is not True
            or source_geometry.get("all_centroid_gates_passed") is not True
            or source_geometry.get("all_separation_gates_passed") is not True
            or source_geometry.get("all_flux_gates_passed") is not True
            or source_geometry.get("gate_passed") is not True
            or any(
                not isinstance(event, Mapping)
                or event.get("resolution_profile_id")
                != C21_RESOLUTION_PROFILE_ID
                or event.get("support_gate_passed") is not True
                or event.get("centroid_gate_passed") is not True
                or event.get("separation_gate_passed") is not True
                or event.get("flux_gate_passed") is not True
                or event.get("source_velocity_m_s") != [0.0, 0.0, -0.1]
                for event in (source_events or [])
            )
        ):
            failures.append("source_emission_geometry")
    declared_particle_count = plan_totals.get("particle_count", 0)
    if (
        isinstance(declared_particle_count, bool)
        or not isinstance(declared_particle_count, int)
        or declared_particle_count < 0
    ):
        declared_particle_count = -1
    expected_runtime_capacity = max(
        required_particle_capacity(), declared_particle_count
    )
    if (
        not isinstance(system, Mapping)
        or system.get("max_velocity_m_s") != MAX_VELOCITY_M_S
        or system.get("smoothing") is not True
        or system.get("smoothing_enabled") is not True
        or system.get("smoothing_strength") != 1.0
        or system.get("anisotropy") is not True
        or system.get("anisotropy_enabled") is not True
        or system.get("anisotropy_scale") != 1.0
        or system.get("ccd") is not True
        or any(
            not _finite_number(system.get(key))
            for key in (
                "particle_contact_offset_m",
                "fluid_rest_offset_m",
                "contact_offset_m",
                "rest_offset_m",
                "solid_rest_offset_m",
                "solver_position_iteration_count",
                "max_neighborhood",
            )
        )
        or not isinstance(system.get("pbd_material_readback"), Mapping)
        or any(
            not _finite_number(system["pbd_material_readback"].get(key))
            for key in (
                "density_kg_m3",
                "friction",
                "cohesion",
                "damping",
                "surface_tension",
                "viscosity",
                "vorticity_confinement",
            )
        )
    ):
        failures.append("particle_postprocessing_contract")
    if isinstance(runtime_configuration, Mapping):
        expected_runtime_configuration = build_runtime_particle_configuration(
            solver_position_iterations=declared_solver_iterations,
            max_particles=expected_runtime_capacity,
            resolution_profile_id=declared_resolution_profile_id,
        ) if (
            declared_solver_iterations > 0
            and declared_particle_count >= 0
            and declared_resolution_profile is not None
        ) else None
        if (
            expected_runtime_configuration is None
            or dict(runtime_configuration) != expected_runtime_configuration
            or any(
                not _finite_number(system.get(key))
                or not math.isclose(
                    float(system[key]),
                    float(expected_runtime_configuration[key]),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                for key in (
                    "particle_contact_offset_m",
                    "fluid_rest_offset_m",
                    "contact_offset_m",
                    "rest_offset_m",
                    "solid_rest_offset_m",
                )
            )
            or system.get("solver_position_iteration_count")
            != declared_solver_iterations
            or system.get(
                "solver_position_iteration_count_requested",
                declared_solver_iterations,
            )
            != declared_solver_iterations
            or system.get("max_neighborhood") != 96
            or system.get("fabric_enabled") is not False
        ):
            failures.append("runtime_particle_configuration")
        collider_readbacks = system.get("collider_collision_readbacks")
        floor_mitigation = system.get("floor_tunneling_mitigation")
        wall_mitigation = system.get("wall_tunneling_mitigation")
        floor_readback = (
            collider_readbacks[0]
            if isinstance(collider_readbacks, list) and collider_readbacks
            else None
        )
        if (
            not isinstance(floor_readback, Mapping)
            or floor_readback.get("path")
            != "/FillStation/FluidmixF2/Colliders/Floor"
            or floor_readback.get("contact_offset_authored") is not True
            or floor_readback.get("rest_offset_authored") is not True
            or not _finite_number(floor_readback.get("contact_offset_m"))
            or not math.isclose(
                float(floor_readback.get("contact_offset_m", math.nan)),
                FLOOR_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not _finite_number(floor_readback.get("rest_offset_m"))
            or not math.isclose(
                float(floor_readback.get("rest_offset_m", math.nan)),
                FLOOR_REST_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not isinstance(floor_mitigation, Mapping)
            or floor_mitigation.get("documented_source")
            != FLOOR_COLLISION_DOCUMENTATION_URL
            or floor_mitigation.get("observed_failure_receipt_sha256")
            != FLOOR_TUNNELING_FAILURE_RECEIPT_SHA256
            or floor_mitigation.get("contact_offset_exceeds_capped_step")
            is not True
            or floor_mitigation.get("readback_passed") is not True
            or not _finite_number(
                floor_mitigation.get("maximum_capped_displacement_per_step_m")
            )
            or not math.isclose(
                float(
                    floor_mitigation.get(
                        "maximum_capped_displacement_per_step_m", math.nan
                    )
                ),
                MAX_VELOCITY_M_S / declared_physics_steps_per_second,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _finite_number(floor_mitigation.get("floor_contact_offset_m"))
            or not math.isclose(
                float(floor_mitigation.get("floor_contact_offset_m", math.nan)),
                FLOOR_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _finite_number(floor_mitigation.get("floor_rest_offset_m"))
            or not math.isclose(
                float(floor_mitigation.get("floor_rest_offset_m", math.nan)),
                FLOOR_REST_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _finite_number(floor_mitigation.get("floor_top_unchanged_m"))
            or not math.isclose(
                float(floor_mitigation.get("floor_top_unchanged_m", math.nan)),
                0.9762065645,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            failures.append("floor_tunneling_mitigation")
        wall_readbacks = (
            collider_readbacks[1:]
            if isinstance(collider_readbacks, list)
            and len(collider_readbacks) == 33
            else None
        )
        expected_wall_paths = [
            f"/FillStation/FluidmixF2/Colliders/Wall{index:02d}"
            for index in range(32)
        ]
        if (
            not isinstance(wall_readbacks, list)
            or any(not isinstance(item, Mapping) for item in wall_readbacks)
            or [item.get("path") for item in wall_readbacks]
            != expected_wall_paths
            or any(
                not isinstance(item, Mapping)
                or item.get("contact_offset_authored") is not True
                or item.get("rest_offset_authored") is not True
                or not _finite_number(item.get("contact_offset_m"))
                or not math.isclose(
                    float(item.get("contact_offset_m", math.nan)),
                    WALL_CONTACT_OFFSET_M,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                or not _finite_number(item.get("rest_offset_m"))
                or not math.isclose(
                    float(item.get("rest_offset_m", math.nan)),
                    WALL_REST_OFFSET_M,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                for item in wall_readbacks
            )
            or not isinstance(wall_mitigation, Mapping)
            or wall_mitigation.get("documented_source")
            != FLOOR_COLLISION_DOCUMENTATION_URL
            or wall_mitigation.get("observed_failure_receipt_sha256")
            != WALL_TUNNELING_FAILURE_RECEIPT_SHA256
            or wall_mitigation.get("wall_count") != 32
            or wall_mitigation.get("contact_offset_exceeds_capped_step")
            is not True
            or wall_mitigation.get("readback_passed") is not True
            or not _finite_number(
                wall_mitigation.get("maximum_capped_displacement_per_step_m")
            )
            or not math.isclose(
                float(
                    wall_mitigation.get(
                        "maximum_capped_displacement_per_step_m", math.nan
                    )
                ),
                MAX_VELOCITY_M_S / declared_physics_steps_per_second,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _finite_number(wall_mitigation.get("wall_contact_offset_m"))
            or not math.isclose(
                float(wall_mitigation.get("wall_contact_offset_m", math.nan)),
                WALL_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _finite_number(wall_mitigation.get("wall_rest_offset_m"))
            or not math.isclose(
                float(wall_mitigation.get("wall_rest_offset_m", math.nan)),
                WALL_REST_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not _finite_number(wall_mitigation.get("wall_thickness_unchanged_m"))
            or not math.isclose(
                float(wall_mitigation.get("wall_thickness_unchanged_m", math.nan)),
                0.001,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            failures.append("wall_tunneling_mitigation")
        gpu = receipt.get("gpu_simulation")
        if (
            not isinstance(gpu, Mapping)
            or not isinstance(gpu.get("position_writeback_initial_readback"), bool)
            or not isinstance(gpu.get("velocity_writeback_initial_readback"), bool)
            or not isinstance(gpu.get("fabric_initial_readback"), bool)
            or gpu.get("position_writeback_enabled_readback") is not True
            or gpu.get("velocity_writeback_enabled_readback") is not True
            or gpu.get("fabric_disabled_readback") is not True
            or gpu.get("position_writeback_restored_readback")
            is not gpu.get("position_writeback_initial_readback")
            or gpu.get("velocity_writeback_restored_readback")
            is not gpu.get("velocity_writeback_initial_readback")
            or gpu.get("fabric_restored_readback")
            is not gpu.get("fabric_initial_readback")
        ):
            failures.append("runtime_writeback_fabric_evidence")
    limitations = receipt.get("render_limitations")
    expected_limitations = (
        COUNT_DRIVEN_RENDER_LIMITATIONS
        if count_driven_body
        else RENDER_LIMITATIONS
    )
    if not isinstance(limitations, Mapping) or any(
        limitations.get(key) is not value
        for key, value in expected_limitations.items()
    ):
        failures.append("render_limitations")
    instancer = receipt.get("point_instancer")
    if (
        not isinstance(instancer, Mapping)
        or not isinstance(instancer.get("path"), str)
        or not instancer.get("path", "").endswith("/FluidParticles/Shared")
        or instancer.get("max_particles")
        != (expected_runtime_capacity if isinstance(runtime_configuration, Mapping) else 2221)
        or instancer.get("ids_stable") is not True
    ):
        failures.append("point_instancer_contract")
    if not _runtime_particle_layer_contract_valid(receipt):
        failures.append("runtime_particle_layer_contract")
    mass_readback = (
        instancer.get("mass_density_readback") if isinstance(instancer, Mapping) else None
    )
    if isinstance(runtime_configuration, Mapping):
        if (
            not isinstance(mass_readback, Mapping)
            or mass_readback.get("mass_api_present") is not False
            or mass_readback.get("particle_mass_argument_kg") is not None
            or mass_readback.get("total_mass_attr_kg") is not None
            or mass_readback.get("total_mass_attr_authored") is not False
            or mass_readback.get("density_attr_kg_m3")
            != runtime_configuration["density_kg_m3"]
            or mass_readback.get("density_attr_authored") is not True
            or not math.isclose(
                float(mass_readback.get("nominal_per_particle_mass_kg", math.nan)),
                runtime_configuration["density_kg_m3"]
                * (2.0 * runtime_configuration["fluid_rest_offset_m"]) ** 3,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            failures.append("particle_mass_density_evidence")
    elif (
        not isinstance(mass_readback, Mapping)
        or mass_readback.get("mass_api_present") is not True
        or not _finite_number(mass_readback.get("total_mass_attr_kg"))
        or not _finite_number(mass_readback.get("density_attr_kg_m3"))
        or not isinstance(mass_readback.get("precedence"), str)
        or not isinstance(mass_readback.get("source"), str)
    ):
        failures.append("particle_mass_density_evidence")
    captures = receipt.get("captures")
    if not isinstance(captures, list):
        failures.append("captures")
    elif any(
        not isinstance(item, Mapping)
        or int(item.get("emitted_particle_count", -1))
        != int(item.get("active_particle_count", -2))
        or int(item.get("active_particle_count", -1))
        != int(item.get("inside_vial_interior_count", -1))
        + int(item.get("in_flight_above_mouth_count", -1))
        + int(item.get("outside_or_escaped_particle_count", -1))
        or int(item.get("outside_vial_interior_count", -1))
        != int(item.get("expected_in_flight_stream_count", -1))
        + int(item.get("leaked_particle_count", -1))
        for item in captures
    ):
        failures.append("containment_partition")
    elif any(item.get("leakage_detected") is True for item in captures if isinstance(item, Mapping)):
        failures.append("particle_leakage")
    if isinstance(captures, list):
        required_volume_records = [
            item
            for item in captures
            if isinstance(item, Mapping)
            and isinstance(item.get("volume_gate"), Mapping)
            and item["volume_gate"].get("required") is True
        ]
        state_gate_failed = not required_volume_records or any(
            item.get("exact_polygon_containment") is not True
            or item.get("finite_state") is not True
            or item.get("contained_z_min_m") is None
            or item.get("contained_z_q05_m") is None
            or item.get("contained_z_q95_m") is None
            or item.get("contained_z_max_m") is None
            or not isinstance(item.get("point_instancer_scale_component_min"), list)
            or len(item.get("point_instancer_scale_component_min", [])) != 3
            or not isinstance(item.get("point_instancer_scale_component_max"), list)
            or len(item.get("point_instancer_scale_component_max", [])) != 3
            or item["volume_gate"].get("state_passed") is not True
            or not isinstance(
                item["volume_gate"].get("smoothed_render_state"), Mapping
            )
            or item["volume_gate"]["smoothed_render_state"].get(
                "particle_bulk_height_m"
            )
            is None
            for item in required_volume_records
        )
        if state_gate_failed:
            failures.append("volume_state_gate")
        visible_gate_failed = not required_volume_records or any(
            not _visible_height_contract_valid(
                item["volume_gate"], require_evidence=True
            )
            for item in required_volume_records
        )
        if visible_gate_failed:
            failures.append("volume_visible_gate")
        if required_volume_records and any(
            not _combined_volume_gate_valid(item["volume_gate"])
            for item in required_volume_records
        ):
            failures.append("volume_gate_contract")

    qualification = receipt.get("qualification")
    if (
        isinstance(qualification, Mapping)
        and qualification.get("first_stage_only") is True
        and (
            qualification.get("volume_state_gate_passed") is not True
            or qualification.get("volume_visible_gate_passed") is not True
            or qualification.get("volume_gate_passed") is not True
        )
    ):
        failures.append("qualification_volume_gate")

    semantics = receipt.get("semantics")
    paired = semantics.get("paired_capture_contract") if isinstance(semantics, Mapping) else None
    if (
        not isinstance(paired, Mapping)
        or paired.get("shared_single_render_pass") is not False
        or paired.get("paired_same_physics_state") is not True
        or paired.get("kept_rgb_uses_source_equivalent_glass") is not True
        or paired.get("semantic_pass_uses_transparent_annotation_flag") is not True
        or paired.get("flag_restored_false_after_semantic_pass") is not True
        or paired.get("state_fields_required_unchanged")
        != list(PAIRED_CAPTURE_STATE_FIELDS)
        + (
            [
                "compositions",
                "body_points",
                "body_extent",
                "body_visibility",
                "body_material_color",
                "stream_visibility",
            ]
            if count_driven_body
            else []
        )
    ):
        failures.append("paired_capture_contract")

    annotation = (
        semantics.get("transparent_annotation_override")
        if isinstance(semantics, Mapping)
        else None
    )
    source = annotation.get("source") if isinstance(annotation, Mapping) else None
    initial = (
        annotation.get("initial_rgb_state") if isinstance(annotation, Mapping) else None
    )
    if (
        not isinstance(annotation, Mapping)
        or annotation.get("capture_mode")
        != "paired_same_physics_state_rgb_then_semantic"
        or annotation.get("documentation_url")
        != TRANSPARENT_ANNOTATION_DOCUMENTATION_URL
        or annotation.get("semantic_pass_value") is not True
        or annotation.get("rgb_and_restore_value") is not False
        or not isinstance(source, Mapping)
        or (
            source.get("value") is not None
            and source.get("value") is not False
        )
        or not isinstance(initial, Mapping)
        or initial.get("composed_value") is not False
        or initial.get("property_stack_strongest_is_session_layer") is not True
        or initial.get("source_equivalent") is not True
        or annotation.get("property_stack_strongest_is_session_layer") is not True
        or annotation.get("authored_after_untouched_reference_capture") is not True
        or annotation.get("source_layer_unchanged_expected") is not True
    ):
        failures.append("transparent_annotation_contract")
    transitions = annotation.get("transitions") if isinstance(annotation, Mapping) else None
    if not isinstance(transitions, list) or not transitions:
        failures.append("transparent_annotation_restore")
    else:
        transition_values = []
        transitions_valid = True
        for transition in transitions:
            if not isinstance(transition, Mapping):
                transitions_valid = False
                continue
            capture_pass = transition.get("capture_pass")
            composed = transition.get("composed_value")
            expected = {
                "kept_rgb_source_equivalent": False,
                "semantic_transparent_annotation": True,
                "restore_after_semantic": False,
            }.get(capture_pass)
            if (
                expected is None
                or transition.get("requested_value") is not expected
                or transition.get("composed_type_name") != "bool"
                or composed is not expected
                or transition.get("property_stack_strongest_is_session_layer")
                is not True
            ):
                transitions_valid = False
            transition_values.append((capture_pass, composed))
        if (
            not transitions_valid
            or not any(
                capture_pass == "semantic_transparent_annotation"
                and composed is True
                for capture_pass, composed in transition_values
            )
            or transition_values[-1] != ("restore_after_semantic", False)
            or sum(
                capture_pass == "semantic_transparent_annotation"
                for capture_pass, _ in transition_values
            )
            != sum(
                capture_pass == "restore_after_semantic"
                for capture_pass, _ in transition_values
            )
        ):
            failures.append("transparent_annotation_restore")

    timing = receipt.get("timing")
    capture_guards = (
        timing.get("capture_guard_records") if isinstance(timing, Mapping) else None
    )
    state_records = paired.get("state_records") if isinstance(paired, Mapping) else None
    if not isinstance(capture_guards, list) or not capture_guards:
        failures.append("paired_capture_guards")
    if not isinstance(state_records, list) or not state_records:
        failures.append("paired_capture_state_drift")
    elif isinstance(capture_guards, list) and any(
        not _paired_state_record_valid(
            record,
            capture_guards,
            require_count_driven_body=count_driven_body,
        )
        for record in state_records
    ):
        failures.append("paired_capture_state_drift")
    if isinstance(state_records, list) and isinstance(transitions, list) and any(
        not all(
            any(
                isinstance(transition, Mapping)
                and transition.get("capture_pass") == capture_pass
                and transition.get("physics_step") == record.get("physics_step")
                and transition.get("phase") == record.get("phase")
                and transition.get("attempt") == record.get("attempt")
                for transition in transitions
            )
            for capture_pass in (
                "kept_rgb_source_equivalent",
                "semantic_transparent_annotation",
                "restore_after_semantic",
            )
        )
        for record in state_records
        if isinstance(record, Mapping)
    ):
        failures.append("transparent_annotation_restore")

    pairs = semantics.get("pairs") if isinstance(semantics, Mapping) else None
    if not isinstance(pairs, list) or not pairs or any(
        not isinstance(pair, Mapping)
        or pair.get("shared_rgb_semantic_capture") is True
        or pair.get("paired_same_physics_state_rgb_semantic_capture") is not True
        or pair.get("paired_capture_state_passed") is not True
        for pair in pairs
    ):
        failures.append("paired_capture_pairs")
    # Isosurface and canonical-body paths must not be part of this receipt.
    if receipt.get("isosurface") is not None or receipt.get("mesh_column") is not None:
        failures.append("retired_render_path_present")
    if count_driven_body:
        failures.extend(_validate_count_driven_body_contract(receipt))
    if (
        isinstance(qualification, Mapping)
        and qualification.get("first_stage_only") is False
    ):
        failures.extend(_validate_full_run_contract(receipt))
    return tuple(dict.fromkeys(failures))


__all__ = [
    "SourceState",
    "F2_MOTION_FALSIFIER_STEP",
    "F2_SAFE_CAPTURE_STRIDE_STEPS",
    "F2_VIDEO_PLAYBACK_FPS",
    "F2_WALL_LIMIT_SECONDS",
    "COUNT_DRIVEN_DODECAGON_AREA_M2",
    "COUNT_DRIVEN_PARTICLE_VOLUME_M3",
    "COUNT_DRIVEN_RENDER_PATH",
    "PERSISTENT_RENDER_STREAM_PATH",
    "COUNT_DRIVEN_RENDER_LIMITATIONS",
    "COUNT_DRIVEN_SOURCE_GEOMETRY_SHA256",
    "COUNT_DRIVEN_SOURCE_TOPOLOGY_SHA256",
    "F2_REQUIRED_ACCEPTANCE_GATES",
    "MAX_VELOCITY_M_S",
    "OPAQUE_BASE_CUTOFF_HEIGHT_M",
    "OPAQUE_BASE_CUTOFF_SOURCE",
    "OPAQUE_BASE_CUTOFF_VOLUME_ML",
    "PAINT_BALL_RECEIPT_SCHEMA_VERSION",
    "SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL",
    "PAIRED_CAPTURE_STATE_FIELDS",
    "RENDER_PRODUCT_CLEANUP_DOCUMENTATION_URL",
    "RENDER_PRODUCT_GATING_DOCUMENTATION_URL",
    "RENDER_PRODUCT_ROTATION_INTERVAL_FRAMES",
    "TRANSPARENT_ANNOTATION_DOCUMENTATION_URL",
    "EMITTER_RENDER_PATH",
    "append_point_instancer_state",
    "append_runtime_particle_state",
    "attach_visible_volume_evidence",
    "apply_nuisance_session_layer",
    "author_nuisance_session_layer",
    "bind_external_nuisance_receipt",
    "build_f2_runtime_acceptance_receipt",
    "build_nuisance_authoring_plan",
    "build_palette_material_path",
    "build_render_state_duplicate_evidence",
    "copy_semantic_effective_render_state",
    "build_runtime_particle_configuration",
    "build_runtime_capture_schedule",
    "build_time_preserving_video_repeat_plan",
    "read_existing_light_inventory",
    "build_source_state",
    "build_prototype_specs",
    "containment_evidence",
    "persist_runtime_particle_snapshot",
    "robust_contained_surface_z_m",
    "build_capture_record",
    "build_count_driven_liquid_body_geometry",
    "signed_mesh_volume_m3",
    "validate_paint_ball_receipt",
]

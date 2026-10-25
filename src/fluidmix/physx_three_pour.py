"""Pure contracts for the bounded one-system PhysX three-pour experiment."""

from __future__ import annotations

from collections.abc import Mapping
import math


FLUID_REST_OFFSET_M = 0.0015
NOMINAL_PARTICLE_VOLUME_ML = 8.0 * FLUID_REST_OFFSET_M**3 * 1_000_000.0
CAMERA_CLIPPING_RANGE = (0.01, 10000.0)
_POUR_ROWS = (
    ("t200", 10, 370, 20),
    ("t400", 10, 370, 20),
    ("t100", 40, 1481, 48),
)
_SUPPORTED_COLLIDERS = {"Cube", "Sphere", "Capsule"}


def build_render_guide_specs() -> tuple[dict[str, object], ...]:
    """Return the fixed render-only empty-vessel guide geometry in metres."""

    return (
        {"name": "BaseXMin", "center_m": (-0.018, -0.003, 0.0), "size_m": (0.001, 0.001, 0.038)},
        {"name": "BaseXMax", "center_m": (0.018, -0.003, 0.0), "size_m": (0.001, 0.001, 0.038)},
        {"name": "BaseZMin", "center_m": (0.0, -0.003, -0.018), "size_m": (0.038, 0.001, 0.001)},
        {"name": "BaseZMax", "center_m": (0.0, -0.003, 0.018), "size_m": (0.038, 0.001, 0.001)},
        {"name": "EdgeXMinZMin", "center_m": (-0.018, 0.073, -0.018), "size_m": (0.001, 0.146, 0.001)},
        {"name": "EdgeXMinZMax", "center_m": (-0.018, 0.073, 0.018), "size_m": (0.001, 0.146, 0.001)},
        {"name": "EdgeXMaxZMin", "center_m": (0.018, 0.073, -0.018), "size_m": (0.001, 0.146, 0.001)},
        {"name": "EdgeXMaxZMax", "center_m": (0.018, 0.073, 0.018), "size_m": (0.001, 0.146, 0.001)},
    )


def build_three_pour_plan() -> dict[str, object]:
    """Return the frozen F2 execution plan without importing Isaac Sim."""

    pours = []
    for ordinal, (tank_id, volume_ml, particle_count, chunk_count) in enumerate(
        _POUR_ROWS, start=1
    ):
        equivalent = particle_count * NOMINAL_PARTICLE_VOLUME_ML
        pours.append(
            {
                "ordinal": ordinal,
                "tank_id": tank_id,
                "programmed_volume_ml": volume_ml,
                "particle_count": particle_count,
                "emission_chunk_count": chunk_count,
                "nominal_mass_convention_equivalent_ml": equivalent,
                "nominal_error_ml": equivalent - volume_ml,
            }
        )
    programmed = sum(row[1] for row in _POUR_ROWS)
    particle_count = sum(row[2] for row in _POUR_ROWS)
    nominal = particle_count * NOMINAL_PARTICLE_VOLUME_ML
    return {
        "schema_version": 1,
        "experiment": "physx_three_pour_f2",
        "seed": 20261025,
        "fluid_rest_offset_m": FLUID_REST_OFFSET_M,
        "nominal_particle_volume_ml": NOMINAL_PARTICLE_VOLUME_ML,
        "pours": pours,
        "totals": {
            "programmed_volume_ml": programmed,
            "particle_count": particle_count,
            "nominal_mass_convention_equivalent_ml": nominal,
            "nominal_error_ml": nominal - programmed,
        },
        "mixing_assumption": (
            "fully_mixed_after_each_settled_pour_uncalibrated_render_parameter"
        ),
        "claims": {
            "physically_displaced_volume": False,
            "absorbance_fitted": False,
            "sim_to_real_validated": False,
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
    return tuple(
        quotient + (1 if index < remainder else 0) for index in range(chunks)
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _append_once(failures: list[str], reason: str) -> None:
    if reason not in failures:
        failures.append(reason)


def validate_three_pour_receipt(receipt: Mapping[str, object]) -> tuple[str, ...]:
    """Return substantive F2 failures; an empty tuple means the run passed."""

    if not isinstance(receipt, Mapping):
        return ("receipt_missing_or_unreadable",)
    failures: list[str] = []
    if receipt.get("schema_version") != 1 or receipt.get("experiment") != "physx_three_pour_f2":
        _append_once(failures, "receipt_identity")
    if receipt.get("status") != "pass":
        _append_once(failures, "run_status_not_pass")

    gpu = _mapping(receipt.get("gpu_simulation"))
    if gpu.get("cuda_device_available") is not True:
        _append_once(failures, "cuda_unavailable")
    if gpu.get("scene_enable_gpu_dynamics") is not True or gpu.get("broadphase") != "GPU":
        _append_once(failures, "gpu_dynamics_not_observed")

    system = _mapping(receipt.get("particle_system"))
    if system.get("count") != 1 or system.get("exactly_one") is not True:
        _append_once(failures, "particle_system_count")
    if system.get("mass_api_count") != 0:
        _append_once(failures, "mass_api_present")
    if system.get("all_particle_sets_target_one_system") is not True:
        _append_once(failures, "particle_system_relation")
    colliders = system.get("collider_types")
    if (
        not isinstance(colliders, list)
        or not colliders
        or any(value not in _SUPPORTED_COLLIDERS for value in colliders)
    ):
        _append_once(failures, "collider_type_unsupported")
    if system.get("pbd_material_binding") is not True or system.get("render_material_binding") is not True:
        _append_once(failures, "material_binding_missing")

    guides = _mapping(receipt.get("render_guides"))
    expected_guide_count = len(build_render_guide_specs())
    guide_paths = guides.get("paths")
    if (
        guides.get("expected_count") != expected_guide_count
        or guides.get("authored_count") != expected_guide_count
        or not isinstance(guide_paths, list)
        or len(guide_paths) != expected_guide_count
    ):
        _append_once(failures, "render_guide_count")
    if guides.get("collision_api_count") != 0:
        _append_once(failures, "render_guide_collision_api")

    camera = _mapping(receipt.get("camera"))
    expected_camera_clipping = list(CAMERA_CLIPPING_RANGE)
    camera_ranges = (
        camera.get("clipping_range_requested"),
        camera.get("clipping_range_authored"),
        camera.get("clipping_range_effective"),
    )
    if (
        camera.get("path") != "/World/Camera"
        or any(
            not isinstance(value, list)
            or value != expected_camera_clipping
            or any(type(component) is not float for component in value)
            for value in camera_ranges
        )
    ):
        _append_once(failures, "camera_clipping_range")

    raw_pours = receipt.get("pours")
    expected_cumulative = 0
    if not isinstance(raw_pours, list) or len(raw_pours) != len(_POUR_ROWS):
        _append_once(failures, "pour_order_or_volume")
        raw_pours = []
    for ordinal, expected in enumerate(_POUR_ROWS, start=1):
        if ordinal > len(raw_pours):
            break
        row = _mapping(raw_pours[ordinal - 1])
        tank_id, volume_ml, particle_count, _ = expected
        expected_cumulative += particle_count
        if (
            row.get("ordinal") != ordinal
            or row.get("tank_id") != tank_id
            or row.get("programmed_volume_ml") != volume_ml
            or row.get("emitted_particle_count") != particle_count
            or row.get("cumulative_particle_count") != expected_cumulative
        ):
            _append_once(failures, "pour_order_or_volume")
        if row.get("finite_positions") is not True:
            _append_once(failures, "particle_output_nonfinite_or_missing")
        if row.get("escaped_particle_count") != 0:
            _append_once(failures, "particles_escaped")
        if row.get("stable") is not True:
            _append_once(failures, "stage_not_stable")
        if row.get("isosurface_nonempty") is not True:
            _append_once(failures, "isosurface_empty")
        before = row.get("used_gpu_memory_mib_before")
        after = row.get("used_gpu_memory_mib_after")
        if not _finite_number(before) or not _finite_number(after):
            _append_once(failures, "memory_sample_missing")
        elif float(after) - float(before) >= 4096 and row.get("memory_recovered_during_settle") is not True:
            _append_once(failures, "stage_memory_not_recovered")

    totals = _mapping(receipt.get("totals"))
    plan_totals = build_three_pour_plan()["totals"]
    if (
        totals.get("programmed_volume_ml") != 60
        or totals.get("emitted_particle_count") != 2221
        or totals.get("final_particle_count") != 2221
        or not _finite_number(totals.get("nominal_mass_convention_equivalent_ml"))
        or not math.isclose(
            float(totals.get("nominal_mass_convention_equivalent_ml", math.nan)),
            float(plan_totals["nominal_mass_convention_equivalent_ml"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not _finite_number(totals.get("nominal_error_ml"))
        or not math.isclose(
            float(totals.get("nominal_error_ml", math.nan)),
            float(plan_totals["nominal_error_ml"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        _append_once(failures, "particle_or_nominal_volume_total")

    isosurface = _mapping(receipt.get("isosurface"))
    if (
        isosurface.get("enabled") is not True
        or not isinstance(isosurface.get("mesh_point_count"), int)
        or int(isosurface.get("mesh_point_count", 0)) <= 0
        or not isinstance(isosurface.get("mesh_face_count"), int)
        or int(isosurface.get("mesh_face_count", 0)) <= 0
        or isosurface.get("mesh_finite_points") is not True
        or isosurface.get("mesh_extent_nonempty") is not True
    ):
        _append_once(failures, "isosurface_empty")

    render = _mapping(receipt.get("render"))
    frame_count = render.get("frame_count")
    frame_hashes = render.get("frame_sha256")
    stage_hashes = render.get("stage_frame_sha256")
    if (
        render.get("video_decodable") is not True
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or render.get("video_decoded_frame_count") != frame_count
        or not isinstance(render.get("video_sha256"), str)
        or len(str(render.get("video_sha256"))) != 64
    ):
        _append_once(failures, "video_not_decodable")
    if (
        render.get("stage_frames_are_video_subsets") is not True
        or not isinstance(frame_hashes, list)
        or len(frame_hashes) != frame_count
        or not isinstance(stage_hashes, list)
        or len(stage_hashes) != 3
        or not set(stage_hashes).issubset(set(frame_hashes))
    ):
        _append_once(failures, "stage_frame_not_subset")

    memory = _mapping(receipt.get("memory"))
    if (
        memory.get("hard_limit_mib") != 40000
        or memory.get("hard_limit_exceeded") is not False
        or not _finite_number(memory.get("peak_used_mib"))
        or float(memory.get("peak_used_mib", math.inf)) >= 40000
        or memory.get("isosurface_time_limit_seconds") != 2700
        or not _finite_number(memory.get("isosurface_enabled_seconds"))
        or float(memory.get("isosurface_enabled_seconds", math.inf)) > 2700
    ):
        _append_once(failures, "memory_limit_exceeded")
    timing = _mapping(receipt.get("timing"))
    if (
        timing.get("wall_limit_seconds") != 5400
        or not _finite_number(timing.get("wall_seconds"))
        or float(timing.get("wall_seconds", math.inf)) > 5400
    ):
        _append_once(failures, "wall_time_exceeded")
    errors = receipt.get("errors")
    if not isinstance(errors, list) or errors:
        _append_once(failures, "run_error_recorded")
    return tuple(failures)

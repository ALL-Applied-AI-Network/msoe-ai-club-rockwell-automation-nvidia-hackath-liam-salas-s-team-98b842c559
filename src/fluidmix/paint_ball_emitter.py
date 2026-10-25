"""CPU contracts for the NVIDIA Paint Ball Emitter F2 render route.

The Isaac-side runner owns the PointInstancer and PhysX attributes.  This
module keeps the append, prototype, containment and receipt rules testable
without importing Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np


PAINT_BALL_RECEIPT_SCHEMA_VERSION = 16
MAX_VELOCITY_M_S = 2.5  # 250 cm/s in ParticlePostProcessingDemo.py at 0.01 m/unit.
EMITTER_RENDER_PATH = "point_instancer_sphere_emitter"
RENDER_LIMITATIONS = {
    "continuous_surface": False,
    "palette_uncalibrated": True,
    "pbd_not_navier_stokes": True,
}
_ARRAY_FIELDS = ("positions", "velocities", "orientations", "scales")


def _copy_rows(value: object, name: str, width: int) -> list[list[float]]:
    rows = np.asarray(value, dtype=np.float64)
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


def containment_evidence(
    positions: object,
    *,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    expected_in_flight_mask: object | None = None,
    cylindrical_center_xy: Sequence[float] | None = None,
    inner_radius_m: float | None = None,
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
    if cylindrical_center_xy is not None or inner_radius_m is not None:
        if cylindrical_center_xy is None or inner_radius_m is None:
            raise ValueError("cylindrical center and radius must be supplied together")
        center = np.asarray(cylindrical_center_xy, dtype=np.float64)
        radius = float(inner_radius_m)
        if center.shape != (2,) or not np.all(np.isfinite(center)) or not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("cylindrical center/radius must be finite and valid")
        inside_xy = np.sum((rows[:, :2] - center[None, :]) ** 2, axis=1) <= (radius + numerical_tolerance_m) ** 2
        inside = inside_xy & inside_z
    else:
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
    }
    if not required.issubset(containment):
        raise ValueError("containment evidence is incomplete")
    return {
        "physics_step": physics_step,
        "phase": phase,
        "active_particle_count": containment["active_particle_count"],
        "inside_vial_interior_count": containment["inside_vial_interior_count"],
        "outside_vial_interior_count": containment["outside_vial_interior_count"],
        "expected_in_flight_stream_count": containment["expected_in_flight_stream_count"],
        "leaked_particle_count": containment["leaked_particle_count"],
        "max_abs_bounds_m": list(containment["max_abs_bounds_m"]),
        "leakage_detected": containment["leakage_detected"],
        "prototype_count": prototype_count,
        "instancer_path": instancer_path,
        "max_velocity_m_s": MAX_VELOCITY_M_S,
        "render_limitations": dict(RENDER_LIMITATIONS),
    }


def validate_paint_ball_receipt(receipt: Mapping[str, object]) -> tuple[str, ...]:
    if not isinstance(receipt, Mapping):
        return ("receipt_missing_or_unreadable",)
    failures: list[str] = []
    if receipt.get("schema_version") != PAINT_BALL_RECEIPT_SCHEMA_VERSION:
        failures.append("receipt_schema")
    if receipt.get("experiment") != "physx_three_pour_f2_paint_ball_emitter":
        failures.append("receipt_identity")
    if receipt.get("render_path") != EMITTER_RENDER_PATH:
        failures.append("render_path")
    if receipt.get("status") not in {"pass", "insufficient_evidence"}:
        failures.append("run_status")
    system = receipt.get("particle_system")
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
    ):
        failures.append("particle_postprocessing_contract")
    limitations = receipt.get("render_limitations")
    if not isinstance(limitations, Mapping) or any(limitations.get(key) is not value for key, value in RENDER_LIMITATIONS.items()):
        failures.append("render_limitations")
    instancer = receipt.get("point_instancer")
    if (
        not isinstance(instancer, Mapping)
        or not isinstance(instancer.get("path"), str)
        or not instancer.get("path", "").endswith("/FluidParticles/Shared")
        or instancer.get("max_particles") != 2221
        or instancer.get("ids_stable") is not True
        or instancer.get("simulation_points_authored") is not True
    ):
        failures.append("point_instancer_contract")
    captures = receipt.get("captures")
    if not isinstance(captures, list):
        failures.append("captures")
    elif any(
        not isinstance(item, Mapping)
        or int(item.get("active_particle_count", -1))
        != int(item.get("inside_vial_interior_count", -1))
        + int(item.get("outside_vial_interior_count", -1))
        or int(item.get("outside_vial_interior_count", -1))
        != int(item.get("expected_in_flight_stream_count", -1))
        + int(item.get("leaked_particle_count", -1))
        for item in captures
    ):
        failures.append("containment_partition")
    elif any(item.get("leakage_detected") is True for item in captures if isinstance(item, Mapping)):
        failures.append("particle_leakage")
    # Isosurface and canonical-body paths must not be part of this receipt.
    if receipt.get("isosurface") is not None or receipt.get("mesh_column") is not None:
        failures.append("retired_render_path_present")
    return tuple(dict.fromkeys(failures))


__all__ = [
    "MAX_VELOCITY_M_S",
    "PAINT_BALL_RECEIPT_SCHEMA_VERSION",
    "EMITTER_RENDER_PATH",
    "append_point_instancer_state",
    "build_prototype_specs",
    "containment_evidence",
    "build_capture_record",
    "validate_paint_ball_receipt",
]

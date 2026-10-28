"""Reusable fail-closed bindings for accepted F2 receipts.

This module is intentionally free of host, commit, and package-path constants so a
source-bound downstream launcher can apply the same runtime-particle contract without
reviving the obsolete C18 assumption that the read-only ``simulationPoints`` attribute
must be authored in the session layer.
"""

from __future__ import annotations

from collections.abc import Mapping

from .paint_ball_emitter import SIMULATION_POINTS_SCHEMA_DOCUMENTATION_URL


AUTHORED_RUNTIME_PROPERTIES = (
    "positions",
    "velocities",
    "orientations",
    "scales",
    "protoIndices",
    "ids",
)


class F2DownstreamBindingError(ValueError):
    """An accepted F2 receipt cannot be safely consumed downstream."""


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def bind_runtime_particle_readback(receipt: object) -> dict[str, object]:
    """Bind authored arrays and documented read-only PhysX position readback.

    ``simulationPoints`` may or may not have a session-layer spec because PhysX owns the
    read-only value. A root-layer spec is always forbidden. Its finite runtime readback
    must agree with the planned particle count, stable ID count, and final containment
    count. The six arrays authored by the renderer remain session-only.
    """

    value = _mapping(receipt)
    plan = _mapping(value.get("plan")) if value is not None else None
    totals = _mapping(plan.get("totals")) if plan is not None else None
    instancer = _mapping(value.get("point_instancer")) if value is not None else None
    layer = (
        _mapping(instancer.get("runtime_layer_evidence"))
        if instancer is not None
        else None
    )
    specs = _mapping(layer.get("property_specs")) if layer is not None else None
    simulation = (
        _mapping(layer.get("simulation_points_runtime_evidence"))
        if layer is not None
        else None
    )
    timing = _mapping(value.get("timing")) if value is not None else None
    captures = value.get("captures") if value is not None else None
    final_capture = (
        _mapping(captures[-1])
        if isinstance(captures, list) and captures
        else None
    )
    expected_count = _count(totals.get("particle_count")) if totals is not None else None
    contained_count = (
        _count(final_capture.get("inside_vial_interior_count"))
        if final_capture is not None
        else None
    )
    session_spec = (
        specs.get("simulationPoints_session_layer_spec")
        if specs is not None
        else None
    )
    authored_timing_names = (
        "positions",
        "velocities",
        "orientations",
        "scales",
        "proto_indices",
        "ids",
    )
    valid = bool(
        value is not None
        and plan is not None
        and totals is not None
        and instancer is not None
        and layer is not None
        and specs is not None
        and simulation is not None
        and timing is not None
        and final_capture is not None
        and expected_count is not None
        and expected_count > 0
        and contained_count == expected_count
        and all(
            specs.get(f"{name}_session_layer_spec") is True
            and specs.get(f"{name}_root_layer_spec") is False
            for name in AUTHORED_RUNTIME_PROPERTIES
        )
        and all(
            timing.get(f"{name}_session_layer_only") is True
            for name in authored_timing_names
        )
        and isinstance(session_spec, bool)
        and specs.get("simulationPoints_root_layer_spec") is False
        and layer.get("authored_session_only_properties")
        == list(AUTHORED_RUNTIME_PROPERTIES)
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
        and simulation.get("containment_particle_count") == expected_count
        and simulation.get("finite_xyz_rows") is True
        and simulation.get("row_counts_match") is True
        and simulation.get("passed") is True
        and timing.get("final_edit_target_is_session") is True
        and timing.get("simulation_points_root_layer_absent") is True
        and timing.get("simulation_points_runtime_readback_passed") is True
    )
    if not valid:
        raise F2DownstreamBindingError("runtime_particle_binding_invalid")
    assert expected_count is not None
    assert contained_count is not None
    assert isinstance(session_spec, bool)
    return {
        "authored_session_only_properties": list(AUTHORED_RUNTIME_PROPERTIES),
        "simulation_points_documented_read_only": True,
        "simulation_points_session_layer_spec": session_spec,
        "simulation_points_root_layer_spec": False,
        "simulation_points_runtime_shape": [expected_count, 3],
        "simulation_points_runtime_row_count": expected_count,
        "expected_particle_count": expected_count,
        "particle_id_count": expected_count,
        "final_contained_particle_count": contained_count,
        "finite_xyz_rows": True,
    }


__all__ = [
    "AUTHORED_RUNTIME_PROPERTIES",
    "F2DownstreamBindingError",
    "bind_runtime_particle_readback",
]

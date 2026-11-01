"""Reusable downstream checks that replace the stale packaged C18 rules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

SIMULATION_POINTS_DOC = (
    "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.2/"
    "dev_guide/schemas/physxschema.html"
)


def _receipt(*, simulation_points_session_spec: bool = False) -> dict[str, object]:
    authored = (
        "positions",
        "velocities",
        "orientations",
        "scales",
        "protoIndices",
        "ids",
    )
    return {
        "plan": {"totals": {"particle_count": 3}},
        "captures": [{"inside_vial_interior_count": 3}],
        "timing": {
            "final_edit_target_is_session": True,
            "positions_session_layer_only": True,
            "velocities_session_layer_only": True,
            "orientations_session_layer_only": True,
            "scales_session_layer_only": True,
            "proto_indices_session_layer_only": True,
            "ids_session_layer_only": True,
            "simulation_points_root_layer_absent": True,
            "simulation_points_runtime_readback_passed": True,
        },
        "point_instancer": {
            "runtime_layer_evidence": {
                "property_specs": {
                    **{
                        f"{name}_session_layer_spec": True for name in authored
                    },
                    **{f"{name}_root_layer_spec": False for name in authored},
                    "simulationPoints_session_layer_spec": (
                        simulation_points_session_spec
                    ),
                    "simulationPoints_root_layer_spec": False,
                },
                "authored_session_only_properties": list(authored),
                "read_only_runtime_property": "simulationPoints",
                "read_only_runtime_property_session_spec_required": False,
                "read_only_runtime_property_root_spec_forbidden": True,
                "read_only_runtime_property_documentation_url": SIMULATION_POINTS_DOC,
                "failed_properties": [],
                "final_edit_target_is_session": True,
                "passed": True,
                "simulation_points_runtime_evidence": {
                    "attribute": "physxParticle:simulationPoints",
                    "documented_read_only": True,
                    "documentation_url": SIMULATION_POINTS_DOC,
                    "runtime_readback_required": True,
                    "runtime_readback_shape": [3, 3],
                    "runtime_readback_row_count": 3,
                    "expected_particle_count": 3,
                    "particle_id_count": 3,
                    "containment_particle_count": 3,
                    "finite_xyz_rows": True,
                    "row_counts_match": True,
                    "passed": True,
                },
            }
        },
    }


@pytest.mark.parametrize("session_spec", [False, True])
def test_read_only_simulation_points_does_not_require_a_session_spec(
    session_spec: bool,
) -> None:
    from twin.cohorts.downstream_binding import bind_runtime_particle_readback

    binding = bind_runtime_particle_readback(
        _receipt(simulation_points_session_spec=session_spec)
    )

    assert binding == {
        "authored_session_only_properties": [
            "positions",
            "velocities",
            "orientations",
            "scales",
            "protoIndices",
            "ids",
        ],
        "simulation_points_documented_read_only": True,
        "simulation_points_session_layer_spec": session_spec,
        "simulation_points_root_layer_spec": False,
        "simulation_points_runtime_shape": [3, 3],
        "simulation_points_runtime_row_count": 3,
        "expected_particle_count": 3,
        "particle_id_count": 3,
        "final_contained_particle_count": 3,
        "finite_xyz_rows": True,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["point_instancer"]["runtime_layer_evidence"][
            "property_specs"
        ].update(simulationPoints_root_layer_spec=True),
        lambda value: value["point_instancer"]["runtime_layer_evidence"][
            "simulation_points_runtime_evidence"
        ].update(finite_xyz_rows=False, passed=False),
        lambda value: value["point_instancer"]["runtime_layer_evidence"][
            "simulation_points_runtime_evidence"
        ].update(runtime_readback_row_count=2, row_counts_match=False, passed=False),
        lambda value: value["point_instancer"]["runtime_layer_evidence"][
            "simulation_points_runtime_evidence"
        ].update(particle_id_count=2, row_counts_match=False, passed=False),
        lambda value: value["captures"][-1].update(inside_vial_interior_count=2),
        lambda value: value["point_instancer"]["runtime_layer_evidence"][
            "property_specs"
        ].update(positions_root_layer_spec=True),
    ],
)
def test_runtime_particle_binding_rejects_root_authorship_nonfinite_or_count_drift(
    mutation,
) -> None:
    from twin.cohorts.downstream_binding import (
        F2DownstreamBindingError,
        bind_runtime_particle_readback,
    )

    receipt = deepcopy(_receipt())
    mutation(receipt)

    with pytest.raises(F2DownstreamBindingError, match="runtime_particle_binding_invalid"):
        bind_runtime_particle_readback(receipt)

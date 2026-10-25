"""CPU contracts for the Paint Ball Emitter PointInstancer route."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from fluidmix.physx_three_pour import build_three_pour_plan

from fluidmix.paint_ball_emitter import (
    MAX_VELOCITY_M_S,
    PAINT_BALL_RECEIPT_SCHEMA_VERSION,
    append_point_instancer_state,
    build_capture_record,
    build_prototype_specs,
    containment_evidence,
    validate_paint_ball_receipt,
)
import fluidmix.paint_ball_emitter as paint_ball_emitter


def test_emitter_append_extends_all_point_instancer_arrays_and_stable_ids():
    previous = {
        "positions": [[1.0, 2.0, 3.0]],
        "velocities": [[0.0, 0.0, -0.1]],
        "orientations": [[1.0, 0.0, 0.0, 0.0]],
        "scales": [[1.0, 1.0, 1.0]],
        "ids": [4],
        "proto_indices": [2],
    }
    updated = append_point_instancer_state(
        previous,
        positions=[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
        velocities=[[0.0, 0.0, -0.1], [0.0, 0.0, -0.1]],
        prototype_index=1,
        particle_ids=[5, 6],
    )
    assert updated["ids"] == [4, 5, 6]
    assert updated["proto_indices"] == [2, 1, 1]
    assert len(updated["positions"]) == len(updated["velocities"]) == 3
    assert len(updated["orientations"]) == len(updated["scales"]) == 3


def test_emitter_append_rejects_id_reordering_or_length_drift():
    with pytest.raises(ValueError, match="stable|length"):
        append_point_instancer_state(
            {"positions": [[0, 0, 0]], "velocities": [[0, 0, 0]], "orientations": [[1, 0, 0, 0]], "scales": [[1, 1, 1]], "ids": [1], "proto_indices": [0]},
            positions=[[1, 1, 1]], velocities=[[0, 0, 0]], prototype_index=0, particle_ids=[1],
        )


def test_prototypes_are_one_sphere_per_fixed_palette_colour():
    specs = build_prototype_specs({"t200": (0.95, 0.38, 0.08), "t400": (0.18, 0.55, 0.75)})
    assert specs == (
        {"prototype_index": 0, "tank_id": "t200", "shape": "Sphere", "semantic_label": "liquid", "color": [0.95, 0.38, 0.08]},
        {"prototype_index": 1, "tank_id": "t400", "shape": "Sphere", "semantic_label": "liquid", "color": [0.18, 0.55, 0.75]},
    )


def test_palette_material_path_uses_a_valid_named_usd_prim_leaf():
    path = paint_ball_emitter.build_palette_material_path(
        "/FillStation/FluidmixF2", 0
    )

    assert path == "/FillStation/FluidmixF2/PaletteMaterials/Palette_00"
    assert path.rsplit("/", 1)[-1][0].isalpha()


def test_containment_record_counts_outside_particles_and_max_bounds():
    positions = np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    evidence = containment_evidence(
        positions,
        lower_bounds=(-1.0, -1.0, 0.0),
        upper_bounds=(1.0, 1.0, 2.0),
    )
    assert evidence["active_particle_count"] == 3
    assert evidence["inside_vial_interior_count"] == 1
    assert evidence["outside_vial_interior_count"] == 2
    assert evidence["max_abs_bounds_m"] == [2.0, 0.0, 3.0]
    assert evidence["leakage_detected"] is True


def test_containment_uses_cylindrical_vial_interior_not_aabb_corners():
    evidence = containment_evidence(
        [[0.9, 0.9, 1.0]],
        lower_bounds=(-1.0, -1.0, 0.0),
        upper_bounds=(1.0, 1.0, 2.0),
        cylindrical_center_xy=(0.0, 0.0),
        inner_radius_m=1.0,
    )
    assert evidence["inside_vial_interior_count"] == 0
    assert evidence["leaked_particle_count"] == 1


def test_capture_and_receipt_explicitly_record_limits_and_max_velocity():
    record = build_capture_record(
        physics_step=120,
        phase="t200-stage",
        containment={"active_particle_count": 1, "inside_vial_interior_count": 1, "outside_vial_interior_count": 0, "expected_in_flight_stream_count": 0, "leaked_particle_count": 0, "max_abs_bounds_m": [0, 0, 1], "leakage_detected": False},
        prototype_count=3,
        instancer_path="/FillStation/FluidmixF2/Particles",
    )
    assert record["physics_step"] == 120
    assert record["prototype_count"] == 3
    assert record["max_velocity_m_s"] == MAX_VELOCITY_M_S
    assert record["render_limitations"]["continuous_surface"] is False
    receipt = {
        "schema_version": PAINT_BALL_RECEIPT_SCHEMA_VERSION,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "status": "pass",
        "render_path": "point_instancer_sphere_emitter",
        "particle_system": {"max_velocity_m_s": MAX_VELOCITY_M_S, "smoothing": True, "smoothing_enabled": True, "smoothing_strength": 1.0, "anisotropy": True, "anisotropy_enabled": True, "anisotropy_scale": 1.0, "ccd": True},
        "point_instancer": {"path": "/FillStation/FluidmixF2/FluidParticles/Shared", "max_particles": 2221, "ids_stable": True, "simulation_points_authored": True},
        "captures": [record],
        "render_limitations": {"continuous_surface": False, "palette_uncalibrated": True, "pbd_not_navier_stokes": True},
    }
    assert validate_paint_ball_receipt(receipt) == ()


def test_120hz_plan_has_frozen_first_stage_and_final_capture_steps():
    plan = build_three_pour_plan(physics_hz=120)
    assert plan["timing"]["physics_hz"] == 120
    assert [pour["stage_capture_step"] for pour in plan["pours"]] == [1412, 2622, 4004]
    assert plan["timing"]["fill_end_step"] == 3968
    assert plan["timing"]["primary_final_capture_step"] == 4004
    assert plan["timing"]["primary_final_offset_s"] == pytest.approx(0.3)


def test_emitter_route_has_no_live_mesh_or_isosurface_lifecycle_call():
    source = (Path(__file__).parents[1] / "scripts" / "physx_three_pour_sequence.py").read_text()
    run_source = source[source.index("def run_sequence") : source.index("\ndef main")]
    assert "build_liquid_proxy_intent(" not in run_source
    assert "copy_exact_mesh_proxy(" not in run_source
    assert "refresh_exact_mesh_proxy(" not in run_source
    assert "_refresh_stage_isosurface_lifecycle(" not in run_source


def test_qualification_capture_contract_is_bounded_and_requires_hidden_target():
    source = (
        Path(__file__).parents[1] / "scripts" / "physx_three_pour_sequence.py"
    ).read_text()
    assert "{706, 1404, 1412}" in source
    assert "sorted(observed_steps) == sorted(int(step) for step in capture_steps)" in source
    assert "len(target_records) == 1" in source
    assert 'target_records[0].get("visible_stream_tanks") == []' in source
    assert "FIRST_VIAL_BOTTOM_ROI_MIN_LIQUID_PIXELS" in source


def test_transparent_vial_override_follows_untouched_reference_capture():
    source = (
        Path(__file__).parents[1] / "scripts" / "physx_three_pour_sequence.py"
    ).read_text()
    run_source = source[source.index("def run_sequence") : source.index("\ndef main")]
    assert (
        run_source.index('failure_phase = "capture_twin_reference"')
        < run_source.index(
            'failure_phase = "author_transparent_vial_annotation_override"'
        )
        < run_source.index('failure_phase = "author_particle_system"')
    )
    assert '"property_stack_strongest_is_session_layer": True' in source

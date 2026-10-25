"""CPU contracts for the twin PointInstancer palette feasibility smoke."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "physx_particle_palette_smoke",
    ROOT / "scripts" / "physx_particle_palette_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_plan_freezes_twin_camera_particle_palette_and_swap():
    plan = smoke.build_smoke_plan()

    assert plan["experiment"] == "physx_particle_palette_smoke"
    assert plan["particle_count"] == 12
    assert plan["particle_system_count"] == 1
    assert plan["prototype_tanks"] == ["t200", "t400", "t100"]
    assert plan["initial_proto_indices"] == [0, 1, 2] * 4
    assert plan["swapped_proto_indices"] == [1, 0, 2] * 4
    assert plan["physics_steps_before_swap"] == 30
    assert plan["physics_steps_during_swap"] == 0
    assert plan["isaac_sim_version"] == "6.0.1"
    assert plan["frozen_time_code"] == 1.0
    assert plan["renderer"] == "PathTracing"
    assert plan["resolution"] == [640, 480]
    assert plan["warmup_updates"] == 60
    assert plan["rt_subframes"] == 64
    assert plan["max_bounces"] == 16
    assert plan["max_specular_and_transmission_bounces"] == 16
    assert plan["async_throttling_enabled"] is False
    assert plan["material_controls"] == {
        "specular_transmission_scattering_depth_m": 0.01,
        "thin_walled": False,
    }
    assert plan["visual_decision"] == "pending_top_level_review"


def test_swap_changes_only_orange_and_blue_indices():
    assert smoke.swap_orange_blue([0, 1, 2, 2, 1, 0]) == [1, 0, 2, 2, 0, 1]
    with pytest.raises(ValueError, match="outside frozen palette"):
        smoke.swap_orange_blue([3])


def passing_receipt():
    return {
        "schema_version": 1,
        "experiment": "physx_particle_palette_smoke",
        "status": "pending_visual",
        "scene_contract": {
            "default_prim": "/FillStation",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "frozen_time_code": 1.0,
            "session_layer_only": True,
            "hidden_source_paths": [smoke.SOURCE_LIQUID_PATH, smoke.SOURCE_STREAM_PATH],
            "custom_camera_count": 0,
            "custom_light_count": 0,
            "custom_backdrop_count": 0,
            "forbidden_authored_paths": [],
        },
        "technical": {
            "isaac_sim_version": "6.0.1.0",
            "cuda_device_available": True,
            "gpu_dynamics": True,
            "particle_system_count": 1,
            "particle_count": 12,
            "fluid_particle_set": True,
            "creation_api": "omni.physx.scripts.particleUtils.add_physx_particleset_pointinstancer",
            "particle_system_target_count": 1,
            "prototype_count": 3,
            "sphere_prototype_count": 3,
            "prototype_material_bindings": 3,
            "material_variants": [
                {
                    "tank_id": tank,
                    "specular_transmission_color": list(color),
                    "specular_transmission_scattering_depth_m": 0.01,
                    "thin_walled": False,
                    "bound": True,
                }
                for tank, color in smoke.TANK_COLORS.items()
            ],
            "collider_count": len(smoke.build_twin_collider_specs()),
            "all_colliders_invisible": True,
            "physics_steps_before_swap": 30,
            "physics_steps_during_swap": 0,
            "finite_positions": True,
            "moved_under_physics": True,
            "maximum_movement_m": 0.001,
            "indices_before": [0, 1, 2] * 4,
            "indices_after": [1, 0, 2] * 4,
            "positions_identical_across_swap": True,
            "position_bytes_identical_across_swap": True,
            "positions_before_sha256": "e" * 64,
            "positions_after_sha256": "e" * 64,
        },
        "render": {
            "renderer": "PathTracing",
            "camera_path": smoke.CAMERA_PATH,
            "resolution": [640, 480],
            "async_throttling_enabled": False,
            "warmup_updates": 60,
            "rt_subframes": 64,
            "max_bounces": 16,
            "max_specular_and_transmission_bounces": 16,
            "before_readable": True,
            "after_readable": True,
            "before_nonconstant": True,
            "after_nonconstant": True,
            "before_sha256": "a" * 64,
            "after_sha256": "b" * 64,
            "before_crop_sha256": "c" * 64,
            "after_crop_sha256": "d" * 64,
            "contact_pair_sha256": "f" * 64,
        },
        "visual_review": {"status": "pending_top_level_review"},
        "errors": [],
    }


def test_receipt_accepts_technical_evidence_but_never_claims_visual_pass():
    receipt = passing_receipt()
    assert smoke.validate_technical_receipt(receipt) == ()
    assert receipt["status"] == "pending_visual"
    assert receipt["visual_review"]["status"] == "pending_top_level_review"


@pytest.mark.parametrize(
    ("mutator", "failure"),
    [
        (lambda r: r["technical"].update(particle_system_count=2), "particle_system_count"),
        (lambda r: r["technical"].update(moved_under_physics=False), "finite_motion"),
        (lambda r: r["technical"].update(positions_identical_across_swap=False), "swap_changed_positions"),
        (lambda r: r["technical"].update(position_bytes_identical_across_swap=False), "swap_changed_position_bytes"),
        (lambda r: r["technical"].update(fluid_particle_set=False), "pointinstancer_contract"),
        (lambda r: r["scene_contract"].update(custom_camera_count=1), "twin_scene_contract"),
        (lambda r: r["render"].update(max_bounces=8), "render_configuration"),
        (lambda r: r["render"].update(before_nonconstant=False), "render_missing_or_constant"),
        (lambda r: r.update(status="pass"), "status_must_remain_pending_visual"),
    ],
)
def test_receipt_rejects_substantive_failures(mutator, failure):
    receipt = deepcopy(passing_receipt())
    mutator(receipt)
    assert failure in smoke.validate_technical_receipt(receipt)


def test_cli_requires_existing_absolute_stage(tmp_path):
    parser = smoke.build_parser()
    relative = parser.parse_args(["--stage", "fill_station.usda", "--out", str(tmp_path / "o")])
    with pytest.raises(ValueError, match="absolute"):
        smoke.validate_args(relative)

    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    valid = parser.parse_args(["--stage", str(stage), "--out", str(tmp_path / "o")])
    smoke.validate_args(valid)

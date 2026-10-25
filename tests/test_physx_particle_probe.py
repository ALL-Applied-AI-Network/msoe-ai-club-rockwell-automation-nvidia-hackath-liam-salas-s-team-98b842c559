"""CPU-safe contract tests for the bounded F1 particle probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "physx_particle_compat_probe", ROOT / "scripts" / "physx_particle_compat_probe.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_parser_accepts_bounded_seed_steps_and_output():
    args = probe.build_parser().parse_args(["--out", "receipt.json", "--steps", "4", "--seed", "19"])

    assert args.out == Path("receipt.json")
    assert args.steps == 4
    assert args.seed == 19


@pytest.mark.parametrize("argv", [["--out", "r.json", "--steps", "0"], ["--out", "r.json", "--steps", "121"]])
def test_parser_rejects_unbounded_steps(argv):
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(argv)


def test_validate_record_requires_explicit_evidence_for_pass():
    record = probe.empty_record(Path("r.json"), steps=2, seed=3)
    record["status"] = "pass"

    errors = probe.validate_record(record)

    assert "pass requires all required evidence" in errors


def test_validate_record_accepts_explicit_unsupported_receipt():
    record = probe.empty_record(Path("r.json"), steps=2, seed=3)
    record["status"] = "unsupported"
    record["errors"] = ["PhysX particle schemas are unavailable"]

    assert probe.validate_record(record) == []


def test_validate_record_rejects_missing_nonempty_isosurface_mesh_evidence():
    record = probe.empty_record(Path("r.json"), steps=2, seed=3)
    record["status"] = "pass"
    record["versions"] = {"isaac": "6.0.1", "kit": "106.5", "physx": "110.1"}
    record["extensions"]["omni.physx"] = True
    record["gpu_simulation"].update(
        scene_enable_gpu_dynamics=True,
        simulation_interface=True,
        stage_attached=True,
        broadphase="GPU",
        cuda_device_available=True,
        update_to_usd=True,
        update_particles_to_usd=True,
    )
    record["particle_set"].update(
        finite_before=True,
        finite_after=True,
        gravity_moved_down=True,
        box_contained_after=True,
        system_relation=True,
        mass_api_present=False,
    )
    record["particle_system"].update(
        exactly_one=True,
        simulation_owner=True,
        pbd_material_binding=True,
        render_material_binding=True,
    )
    record["isosurface"].update(api_present=True, enabled=True, prim_present=True)
    record["timing"]["simulation_route"] = "attach_stage_simulate_fetch_results"

    errors = probe.validate_record(record)

    assert "pass requires nonempty finite isosurface mesh evidence" in errors


def test_validate_record_accepts_complete_mesh_and_route_evidence():
    record = probe.empty_record(Path("r.json"), steps=2, seed=3)
    record["status"] = "pass"
    record["versions"] = {"isaac": "6.0.1", "kit": "106.5", "physx": "110.1"}
    record["extensions"]["omni.physx"] = True
    record["gpu_simulation"].update(
        scene_enable_gpu_dynamics=True,
        simulation_interface=True,
        stage_attached=True,
        broadphase="GPU",
        cuda_device_available=True,
        update_to_usd=True,
        update_particles_to_usd=True,
    )
    record["particle_set"].update(
        finite_before=True,
        finite_after=True,
        gravity_moved_down=True,
        box_contained_after=True,
        system_relation=True,
        mass_api_present=False,
    )
    record["particle_system"].update(
        exactly_one=True,
        simulation_owner=True,
        pbd_material_binding=True,
        render_material_binding=True,
    )
    record["isosurface"].update(
        api_present=True,
        enabled=True,
        prim_present=True,
        mesh_is_mesh=True,
        mesh_point_count=12,
        mesh_face_count=4,
        mesh_finite_points=True,
        mesh_extent_nonempty=True,
    )
    record["timing"]["simulation_route"] = "attach_stage_simulate_fetch_results"
    for key in record["schema_api"]:
        record["schema_api"][key] = True

    assert probe.validate_record(record) == []


def test_main_persists_failed_receipt_before_fast_shutdown(tmp_path, monkeypatch):
    receipt_path = tmp_path / "receipt.json"
    observed = []

    class FakeOwnedApp:
        def close(self, *, exit_code):
            observed.append(
                {
                    "receipt_existed_at_close": receipt_path.is_file(),
                    "exit_code": exit_code,
                }
            )

    record = probe.empty_record(receipt_path, steps=8, seed=20261025)
    record["status"] = "unsupported"
    record["errors"] = ["deliberate fixture failure"]
    monkeypatch.setattr(
        probe,
        "_start_or_reuse_kit",
        lambda: (object(), FakeOwnedApp()),
    )
    monkeypatch.setattr(probe, "run_probe", lambda _args: record)

    result = probe.main(
        ["--out", str(receipt_path), "--steps", "8", "--seed", "20261025"]
    )

    assert result == 2
    assert observed == [{"receipt_existed_at_close": True, "exit_code": 2}]
    assert receipt_path.is_file()


def test_preview_surface_uses_a_real_shader_output_connection():
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/World/PBDMaterial")

    shader = probe._author_preview_surface(stage, material, Sdf, Gf, UsdShade)

    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    connected_source = material.GetSurfaceOutput().GetConnectedSource()
    assert connected_source is not None
    assert connected_source[0].GetPrim().GetPath() == shader.GetPrim().GetPath()
    assert str(connected_source[1]) == "surface"


def test_version_info_resolves_the_enabled_versioned_physx_extension_id():
    class FakeManager:
        def get_enabled_extension_id(self, name):
            assert name == "omni.physx"
            return "omni.physx-110.4.4"

        def get_extension_dict(self, extension_id):
            assert extension_id == "omni.physx-110.4.4"
            return {"package": {"version": "110.4.4"}}

    class FakeApp:
        def get_build_version(self):
            return "110.4.0"

        def get_extension_manager(self):
            return FakeManager()

    assert probe._version_info(FakeApp())["physx"] == "110.4.4"


def test_version_info_falls_back_to_versioned_extension_id_when_metadata_omits_version():
    class FakeManager:
        def get_enabled_extension_id(self, _name):
            return "omni.physx-110.4.4"

        def get_extension_dict(self, _extension_id):
            return None

    class FakeApp:
        def get_build_version(self):
            return "110.4.0"

        def get_extension_manager(self):
            return FakeManager()

    assert probe._version_info(FakeApp())["physx"] == "110.4.4"

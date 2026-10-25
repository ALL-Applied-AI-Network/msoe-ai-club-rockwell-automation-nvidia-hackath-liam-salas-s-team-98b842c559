"""CPU contracts for the Isaac 6.2 physics-only render-decoupling probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "physx_manual_step_probe",
    ROOT / "scripts" / "physx_manual_step_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_probe_plan_freezes_manual_physics_and_one_explicit_capture():
    plan = probe.build_probe_plan(seed=20261025)

    assert plan == {
        "schema_version": 1,
        "seed": 20261025,
        "physics_hz": 120,
        "physics_dt_s": 1.0 / 120.0,
        "manual_steps": 120,
        "emission_steps": [1, 13, 25, 37, 49, 61, 73, 85, 97, 109],
        "particles_per_emission": 4,
        "final_particle_count": 40,
        "capture_step": 120,
        "rt_subframes": 64,
        "update_fabric": False,
        "render_updates_between_captures": False,
        "capture": {
            "delta_time_s": 0.0,
            "wait_for_render": True,
            "reset_renderer_accumulation": True,
            "render_product_updates_enabled_only_for_capture": True,
        },
        "documentation": [
            "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_sdg_workflows.html",
            "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_getting_started.html",
            "https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html",
        ],
    }


def _passing_receipt():
    plan = probe.build_probe_plan(seed=20261025)
    before = "1" * 64
    after = "1" * 64
    layer_records = [
        {
            "identifier": "fill_station.usda",
            "resolved_path": "/external/fill_station.usda",
            "bytes": 123,
            "sha256": "5" * 64,
        }
    ]
    return {
        "schema_version": 1,
        "experiment": "physx_manual_step_render_decoupling_probe",
        "status": "pass",
        "plan": plan,
        "source": {
            "stage_sha256": "5" * 64,
            "layers_before": deepcopy(layer_records),
            "layers_after": deepcopy(layer_records),
            "source_layers_unchanged": True,
        },
        "scene": {
            "default_prim": "/FillStation",
            "meters_per_unit": 1.0,
            "up_axis": "Z",
            "camera_path": probe.CAMERA_PATH,
            "clipping_range": list(probe.CAMERA_CLIPPING_RANGE),
            "source_liquid_hidden": True,
            "source_stream_hidden": True,
            "transparent_annotation_override": {"session_layer_only": True},
        },
        "runtime": {
            "isaac_sim": "6.2.0.0",
            "kit": "110.4.0+feature",
            "physx": "110.4.4",
            "simulation_manager_method": "SimulationManager.step",
            "update_fabric": False,
            "gpu_dynamics": True,
            "cuda_device_available": True,
            "usd_writeback_settings": {
                "update_to_usd": True,
                "update_particles_to_usd": True,
                "update_velocities_to_usd": True,
            },
        },
        "physics_initialization": {
            "initialized_with_empty_particle_set": True,
            "callbacks_registered_after_initialization": True,
            "baseline_num_physics_steps": 2,
            "baseline_simulation_time_s": 1.0 / 60.0,
        },
        "physics": {
            "manual_step_calls": 120,
            "callbacks_observed": 120,
            "callback_indices": list(range(1, 121)),
            "callback_dt_s": [1.0 / 120.0] * 120,
            "num_physics_steps_delta": 120,
            "simulation_time_delta_s": 1.0,
            "manual_loop_seconds": 0.05,
        },
        "emission": {
            "one_shared_point_instancer": True,
            "point_instancer_path": f"{probe.PROBE_NAMESPACE}/Particles/Shared",
            "max_particles": 40,
            "append_event_steps": plan["emission_steps"],
            "append_event_counts": [4] * 10,
            "final_particle_count": 40,
            "final_particle_ids": list(range(40)),
            "stable_ids_each_step": True,
            "equal_row_counts_each_step": True,
            "particle_identity_checks": 120,
            "row_count_checks": 120,
        },
        "motion": {
            "finite_positions": True,
            "finite_simulation_points": True,
            "finite_velocities": True,
            "maximum_displacement_m": 0.03,
            "finite_state_checks": 120,
        },
        "render": {
            "render_updates_disabled_between_captures": True,
            "renderer_accumulation_reset": True,
            "capture_delta_time_s": 0.0,
            "capture_wait_for_render": True,
            "capture_rt_subframes": 64,
            "capture_step": 120,
            "capture_seconds": 1.5,
            "explicit_capture_count": 1,
            "rgb_readable": True,
            "semantic_readable": True,
            "liquid_mask_readable": True,
            "raster_shape": [480, 640],
            "liquid_pixel_count": 64,
            "semantic_unique_ids": [0, 1, 2],
            "liquid_semantic_ids": [2],
            "semantic_id_to_class_labels": {
                "0": ["background"],
                "1": ["vial"],
                "2": ["liquid"],
            },
        },
        "same_state_capture": {
            "callback_count_unchanged": True,
            "num_physics_steps_unchanged": True,
            "simulation_time_unchanged": True,
            "positions_sha256_before": before,
            "positions_sha256_after": after,
            "velocities_sha256_before": before,
            "velocities_sha256_after": after,
            "ids_sha256_before": before,
            "ids_sha256_after": after,
        },
        "artifacts": {
            "rgb_path": "rgb.png",
            "rgb_sha256": "2" * 64,
            "semantic_path": "semantic.npy",
            "semantic_sha256": "3" * 64,
            "liquid_mask_path": "liquid.png",
            "liquid_mask_sha256": "4" * 64,
        },
        "errors": [],
    }


def test_receipt_validator_accepts_only_the_complete_manual_step_contract():
    assert probe.validate_probe_receipt(_passing_receipt()) == ()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row["source"].update(source_layers_unchanged=False), "source_layers_changed"),
        (lambda row: row["source"].update(layers_after=[]), "source_layer_evidence_invalid"),
        (lambda row: row["scene"].update(default_prim="/World"), "scene_contract_mismatch"),
        (lambda row: row["runtime"].update(kit="110.3.0"), "runtime_contract_mismatch"),
        (lambda row: row["runtime"].update(physx=None), "runtime_contract_mismatch"),
        (
            lambda row: row["runtime"]["usd_writeback_settings"].update(
                update_velocities_to_usd=False
            ),
            "runtime_contract_mismatch",
        ),
        (lambda row: row["physics"].update(callbacks_observed=119), "callback_count_mismatch"),
        (lambda row: row["physics"].update(callback_dt_s=[1.0 / 60.0] * 120), "callback_dt_mismatch"),
        (lambda row: row["physics"].update(num_physics_steps_delta=119), "physics_step_delta_mismatch"),
        (lambda row: row["physics"].update(manual_loop_seconds=None), "timing_evidence_missing"),
        (lambda row: row["emission"].update(append_event_steps=[1]), "emission_schedule_mismatch"),
        (lambda row: row["emission"].update(point_instancer_path="/Wrong"), "shared_point_instancer_missing"),
        (lambda row: row["emission"].update(max_particles=39), "shared_point_instancer_missing"),
        (lambda row: row["emission"].update(final_particle_ids=list(range(39))), "particle_identity_mismatch"),
        (lambda row: row["emission"].update(row_count_checks=119), "particle_identity_mismatch"),
        (lambda row: row["motion"].update(maximum_displacement_m=0.0), "particle_motion_missing"),
        (lambda row: row["motion"].update(finite_simulation_points=False), "particle_motion_missing"),
        (lambda row: row["motion"].update(finite_state_checks=119), "particle_motion_missing"),
        (lambda row: row["render"].update(render_updates_disabled_between_captures=False), "render_not_decoupled"),
        (lambda row: row["render"].update(capture_step=119), "render_not_decoupled"),
        (lambda row: row["render"].update(explicit_capture_count=2), "render_not_decoupled"),
        (lambda row: row["render"].update(capture_seconds=None), "timing_evidence_missing"),
        (lambda row: row["render"].update(liquid_pixel_count=0), "semantic_liquid_missing"),
        (lambda row: row["render"].update(liquid_semantic_ids=[]), "semantic_mapping_missing"),
        (lambda row: row["same_state_capture"].update(callback_count_unchanged=False), "capture_advanced_physics"),
        (lambda row: row["same_state_capture"].update(positions_sha256_after="9" * 64), "capture_changed_particle_state"),
        (lambda row: row["artifacts"].update(rgb_sha256=None), "artifact_hash_missing"),
        (lambda row: row["artifacts"].update(rgb_path=None), "artifact_path_missing"),
    ],
)
def test_receipt_validator_fails_closed_on_each_substantive_check(mutation, reason):
    receipt = deepcopy(_passing_receipt())
    mutation(receipt)

    assert reason in probe.validate_probe_receipt(receipt)


def test_plan_rejects_boolean_seed():
    with pytest.raises(TypeError, match="seed"):
        probe.build_probe_plan(seed=True)
    with pytest.raises(ValueError, match="frozen"):
        probe.build_probe_plan(seed=1)


def test_parser_freezes_seed_and_requires_stage_and_output():
    args = probe.build_parser().parse_args(
        [
            "--out",
            "probe-output",
            "--stage",
            "fill_station.usda",
            "--seed",
            "20261025",
        ]
    )

    assert args.out == Path("probe-output")
    assert args.stage == Path("fill_station.usda")
    assert args.seed == 20261025
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(
            [
                "--out",
                "probe-output",
                "--stage",
                "fill_station.usda",
                "--seed",
                "1",
            ]
        )


def test_runtime_source_uses_documented_manual_step_and_explicit_capture_pattern():
    source = (ROOT / "scripts" / "physx_manual_step_probe.py").read_text(
        encoding="utf-8"
    )

    assert "SimulationManager.step(steps=1, update_fabric=False)" in source
    assert "SimulationContext.step" not in source
    assert "hydra_texture.set_updates_enabled(False)" in source
    assert "reset_renderer_accumulation()" in source
    assert "delta_time=0.0" in source
    assert "wait_for_render=True" in source
    assert "rt_subframes=RT_SUBFRAMES" in source
    assert "f2._version_info(omni.kit.app.get_app())" in source
    manual_loop = source[
        source.index("for step in range(1, MANUAL_STEPS + 1):") : source.index(
            'receipt["physics"]["manual_loop_seconds"]'
        )
    ]
    assert "app.update()" not in manual_loop
    assert "rep.orchestrator.step" not in manual_loop
    assert "timeline.play" not in source


def test_state_snapshot_reads_render_and_simulation_usd_rows_separately():
    class Attribute:
        def __init__(self, value):
            self.value = value

        def Get(self):
            return self.value

    class Instancer:
        def GetPositionsAttr(self):
            return Attribute([(1.0, 2.0, 3.0)])

        def GetVelocitiesAttr(self):
            return Attribute([(0.0, 0.0, -1.0)])

        def GetIdsAttr(self):
            return Attribute([7])

    class PointSet:
        def GetSimulationPointsAttr(self):
            return Attribute([(4.0, 5.0, 6.0)])

    snapshot = probe._state_snapshot(Instancer(), PointSet())

    assert snapshot["positions"].tolist() == [[1.0, 2.0, 3.0]]
    assert snapshot["simulation_positions"].tolist() == [[4.0, 5.0, 6.0]]
    assert snapshot["velocities"].tolist() == [[0.0, 0.0, -1.0]]
    assert snapshot["ids"] == [7]
    assert snapshot["positions_sha256"] == probe._array_sha256([[1.0, 2.0, 3.0]])


def test_receipt_validator_handles_missing_evidence_without_throwing():
    assert probe.validate_probe_receipt(None) == ("receipt_not_mapping",)
    failures = probe.validate_probe_receipt({})
    assert "plan_missing_or_invalid" in failures
    assert "physics_evidence_missing" in failures
    assert "artifact_hash_missing" in failures


def test_main_records_runtime_and_close_tracebacks_without_isaac(
    tmp_path, monkeypatch
):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "probe-output"

    class FakeApp:
        def __init__(self, _config):
            pass

        def close(self):
            assert (output / "receipt.json").is_file()
            raise RuntimeError("close failed")

    fake_isaacsim = ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = FakeApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)

    def fail_probe(_args, _app, _receipt):
        raise ValueError("probe failed")

    monkeypatch.setattr(probe, "run_probe", fail_probe)

    assert probe.main(["--stage", str(stage), "--out", str(output)]) == 2
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "insufficient_evidence"
    assert [record["phase"] for record in receipt["errors"]] == [
        "preflight",
        "simulation_app_close",
    ]
    assert "ValueError: probe failed" in receipt["errors"][0]["traceback"]
    assert "RuntimeError: close failed" in receipt["errors"][1]["traceback"]

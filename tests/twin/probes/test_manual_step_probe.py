"""CPU contracts for the Isaac 6.2 physics-only render-decoupling probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "physx_manual_step_probe",
    ROOT / "cli/twin/probes" / "physx_manual_step_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_probe_plan_freezes_manual_physics_and_paired_explicit_capture():
    plan = probe.build_probe_plan(seed=20261025)

    assert plan == {
        "schema_version": 2,
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
        "runtime_append_ingestion": {
            "method": "one_stopped_timeline_app_update_after_each_append",
            "updates_per_append": 1,
            "expected_update_count": 10,
            "render_product_updates_enabled": False,
            "unchanged_fields": [
                "callback_count",
                "num_physics_steps",
                "simulation_time_s",
                "timeline_time_s",
            ],
            "timeline_playing": False,
        },
        "capture": {
            "delta_time_s": 0.0,
            "wait_for_render": True,
            "reset_renderer_accumulation_each_pass": True,
            "render_product_updates_enabled_only_during_passes": True,
            "capture_mode": "paired_same_physics_state_rgb_then_semantic",
            "shared_single_render_pass": False,
            "render_passes": [
                {
                    "capture_pass": "kept_rgb_source_equivalent",
                    "annotation_value": False,
                    "artifact": "rgb",
                },
                {
                    "capture_pass": "semantic_transparent_annotation",
                    "annotation_value": True,
                    "artifact": "semantic",
                },
            ],
            "restore_annotation_value": False,
        },
        "documentation": [
            "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_sdg_workflows.html",
            "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_getting_started.html",
            "https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html",
            "https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotations_with_transparency.html",
        ],
    }


def _passing_receipt():
    plan = probe.build_probe_plan(seed=20261025)
    paired_state = _paired_state(count=40)
    paired_evidence = probe._paired_capture_evidence(
        {
            "before_rgb": deepcopy(paired_state),
            "after_rgb": deepcopy(paired_state),
            "after_semantic": deepcopy(paired_state),
        }
    )
    layer_records = [
        {
            "identifier": "fill_station.usda",
            "resolved_path": "/external/fill_station.usda",
            "bytes": 123,
            "sha256": "5" * 64,
        }
    ]
    ingestion_records = []
    for index, emission_step in enumerate(plan["emission_steps"]):
        state = {
            "callback_count": 12 * index,
            "num_physics_steps": 2 + 12 * index,
            "simulation_time_s": 1.0 / 60.0 + 0.1 * index,
            "timeline_time_s": 1.0 / 24.0,
            "timeline_playing": False,
        }
        ingestion_records.append(
            probe._ingestion_update_evidence(
                emission_step=emission_step,
                before=deepcopy(state),
                after=deepcopy(state),
            )
        )
    return {
        "schema_version": 2,
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
            "transparent_annotation_override": {
                "capture_mode": "paired_same_physics_state_rgb_then_semantic",
                "semantic_pass_value": True,
                "rgb_and_restore_value": False,
                "initial_rgb_state": {
                    "composed_value": False,
                    "source_equivalent": True,
                    "property_stack_strongest_is_session_layer": True,
                },
                "transitions": [
                    {
                        "capture_pass": "kept_rgb_source_equivalent",
                        "requested_value": False,
                        "composed_value": False,
                        "property_stack_strongest_is_session_layer": True,
                    },
                    {
                        "capture_pass": "semantic_transparent_annotation",
                        "requested_value": True,
                        "composed_value": True,
                        "property_stack_strongest_is_session_layer": True,
                    },
                    {
                        "capture_pass": "restore_after_semantic",
                        "requested_value": False,
                        "composed_value": False,
                        "property_stack_strongest_is_session_layer": True,
                    },
                ],
                "property_stack_strongest_is_session_layer": True,
                "authored_after_untouched_reference_capture": False,
                "restore_verified": True,
            },
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
        "runtime_append_ingestion": {
            "method": "one_stopped_timeline_app_update_after_each_append",
            "app_update_count": 10,
            "render_product_updates_enabled": False,
            "records": ingestion_records,
            "all_updates_preserved_physics_state": True,
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
            "render_product_updates_disabled_outside_passes": True,
            "renderer_accumulation_reset_count": 2,
            "capture_delta_time_s": 0.0,
            "capture_wait_for_render": True,
            "capture_rt_subframes": 64,
            "capture_step": 120,
            "paired_capture_seconds": 1.5,
            "paired_render_pass_count": 2,
            "shared_single_render_pass": False,
            "capture_mode": "paired_same_physics_state_rgb_then_semantic",
            "capture_passes": [
                {
                    "pass_index": 1,
                    "capture_pass": "kept_rgb_source_equivalent",
                    "annotation_value": False,
                    "renderer_accumulation_reset": True,
                    "render_product_updates_enabled": True,
                    "completed": True,
                    "seconds": 0.7,
                },
                {
                    "pass_index": 2,
                    "capture_pass": "semantic_transparent_annotation",
                    "annotation_value": True,
                    "renderer_accumulation_reset": True,
                    "render_product_updates_enabled": True,
                    "completed": True,
                    "seconds": 0.8,
                },
            ],
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
        "same_state_capture": paired_evidence,
        "artifacts": {
            "rgb_path": "rgb.png",
            "rgb_sha256": "2" * 64,
            "rgb_capture_pass": "kept_rgb_source_equivalent",
            "semantic_path": "semantic.npy",
            "semantic_sha256": "3" * 64,
            "semantic_capture_pass": "semantic_transparent_annotation",
            "liquid_mask_path": "liquid.png",
            "liquid_mask_sha256": "4" * 64,
            "liquid_mask_capture_pass": "semantic_transparent_annotation",
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
        (
            lambda row: row["scene"]["transparent_annotation_override"].update(
                restore_verified=False
            ),
            "transparent_annotation_restore",
        ),
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
        (
            lambda row: row["runtime_append_ingestion"]["records"][0][
                "after"
            ].update(callback_count=1),
            "runtime_append_ingestion_contract",
        ),
        (
            lambda row: row["runtime_append_ingestion"].update(
                render_product_updates_enabled=True
            ),
            "runtime_append_ingestion_contract",
        ),
        (lambda row: row["emission"].update(append_event_steps=[1]), "emission_schedule_mismatch"),
        (lambda row: row["emission"].update(point_instancer_path="/Wrong"), "shared_point_instancer_missing"),
        (lambda row: row["emission"].update(max_particles=39), "shared_point_instancer_missing"),
        (lambda row: row["emission"].update(final_particle_ids=list(range(39))), "particle_identity_mismatch"),
        (lambda row: row["emission"].update(row_count_checks=119), "particle_identity_mismatch"),
        (lambda row: row["motion"].update(maximum_displacement_m=0.0), "particle_motion_missing"),
        (lambda row: row["motion"].update(finite_simulation_points=False), "particle_motion_missing"),
        (lambda row: row["motion"].update(finite_state_checks=119), "particle_motion_missing"),
        (
            lambda row: row["render"].update(
                render_product_updates_disabled_outside_passes=False
            ),
            "render_not_decoupled",
        ),
        (lambda row: row["render"].update(capture_step=119), "render_not_decoupled"),
        (lambda row: row["render"].update(paired_render_pass_count=1), "render_not_decoupled"),
        (lambda row: row["render"].update(shared_single_render_pass=True), "render_not_decoupled"),
        (lambda row: row["render"].update(paired_capture_seconds=None), "timing_evidence_missing"),
        (lambda row: row["render"].update(liquid_pixel_count=0), "semantic_liquid_missing"),
        (lambda row: row["render"].update(liquid_semantic_ids=[]), "semantic_mapping_missing"),
        (
            lambda row: row["same_state_capture"].update(
                callback_counts=[120, 120, 121]
            ),
            "capture_advanced_physics",
        ),
        (
            lambda row: row["same_state_capture"]["hashes"].update(
                positions=["1" * 64, "1" * 64, "9" * 64]
            ),
            "capture_changed_particle_state",
        ),
        (lambda row: row["artifacts"].update(rgb_sha256=None), "artifact_hash_missing"),
        (lambda row: row["artifacts"].update(rgb_path=None), "artifact_path_missing"),
        (
            lambda row: row["artifacts"].update(
                rgb_capture_pass="semantic_transparent_annotation"
            ),
            "artifact_pass_mismatch",
        ),
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


def test_runtime_source_uses_documented_manual_step_and_a27_paired_capture_pattern():
    source = (ROOT / "cli/twin/probes" / "physx_manual_step_probe.py").read_text(
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
    assert "f2._author_transparent_vial_annotation_override" not in source
    assert "f2._prepare_transparent_vial_annotation_control" in source
    assert "f2._set_transparent_vial_annotation_state" in source
    assert '"shared_single_render_pass": False' in source
    assert 'render.get("paired_render_pass_count") != 2' in source
    render_attach = source[
        source.index("rgb_annotator.attach") : source.index(
            'receipt["runtime"]["failure_phase"] = "initialize_manual_physics"'
        )
    ]
    assert render_attach.index("set_updates_enabled(False)") < render_attach.index(
        "for _ in range(60):"
    )
    manual_loop = source[
        source.index("for step in range(1, MANUAL_STEPS + 1):") : source.index(
            'receipt["physics"]["manual_loop_seconds"]'
        )
    ]
    assert manual_loop.count("app.update()") == 1
    emission_branch = manual_loop[
        manual_loop.index("if step in EMISSION_STEPS:") : manual_loop.index(
            "callbacks_before = len(callbacks)"
        )
    ]
    assert emission_branch.count("app.update()") == 1
    assert manual_loop.index("app.update()") < manual_loop.index(
        "SimulationManager.step(steps=1, update_fabric=False)"
    )
    assert "timeline.is_playing()" in emission_branch
    assert "hydra_texture.set_updates_enabled(False)" in emission_branch
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

        def GetProtoIndicesAttr(self):
            return Attribute([3])

    class PointSet:
        def GetSimulationPointsAttr(self):
            return Attribute([(4.0, 5.0, 6.0)])

    snapshot = probe._state_snapshot(Instancer(), PointSet())

    assert snapshot["positions"].tolist() == [[1.0, 2.0, 3.0]]
    assert snapshot["simulation_positions"].tolist() == [[4.0, 5.0, 6.0]]
    assert snapshot["velocities"].tolist() == [[0.0, 0.0, -1.0]]
    assert snapshot["ids"] == [7]
    assert snapshot["proto_indices"] == [3]
    assert snapshot["positions_sha256"] == probe._array_sha256([[1.0, 2.0, 3.0]])


def test_ingestion_update_evidence_requires_stopped_timeline_and_no_physics_advance():
    state = {
        "callback_count": 12,
        "num_physics_steps": 14,
        "simulation_time_s": 0.11666666666666667,
        "timeline_time_s": 1.0 / 24.0,
        "timeline_playing": False,
    }

    evidence = probe._ingestion_update_evidence(
        emission_step=13,
        before=deepcopy(state),
        after=deepcopy(state),
    )

    assert evidence == {
        "schema_version": 1,
        "emission_step": 13,
        "app_update_calls": 1,
        "render_product_updates_enabled": False,
        "before": state,
        "after": state,
        "passed": True,
        "failed_predicates": [],
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("callback_count", 13),
        ("num_physics_steps", 15),
        ("simulation_time_s", 0.2),
        ("timeline_time_s", 0.1),
        ("timeline_playing", True),
    ],
)
def test_ingestion_update_evidence_names_state_advanced_by_app_update(
    field, replacement
):
    before = {
        "callback_count": 12,
        "num_physics_steps": 14,
        "simulation_time_s": 0.11666666666666667,
        "timeline_time_s": 1.0 / 24.0,
        "timeline_playing": False,
    }
    after = deepcopy(before)
    after[field] = replacement

    evidence = probe._ingestion_update_evidence(
        emission_step=13,
        before=before,
        after=after,
    )

    assert evidence["passed"] is False
    assert field in evidence["failed_predicates"]


def _paired_state(count=2):
    positions = [
        [float(index), float(index) + 0.25, float(index) + 0.5]
        for index in range(count)
    ]
    return {
        "callback_count": 120,
        "num_physics_steps": 122,
        "simulation_time_s": 1.0166666666666666,
        "timeline_time_s": 1.0 / 24.0,
        "timeline_playing": False,
        "camera_path": probe.CAMERA_PATH,
        "positions": deepcopy(positions),
        "simulation_positions": deepcopy(positions),
        "velocities": [[0.0, 0.0, -0.1] for _ in range(count)],
        "ids": list(range(count)),
        "proto_indices": [0] * count,
    }


def test_paired_capture_evidence_proves_both_render_passes_preserve_exact_state():
    state = _paired_state()

    evidence = probe._paired_capture_evidence(
        {
            "before_rgb": deepcopy(state),
            "after_rgb": deepcopy(state),
            "after_semantic": deepcopy(state),
        }
    )

    assert evidence["passed"] is True
    assert evidence["failed_predicates"] == []
    assert evidence["state_labels"] == [
        "before_rgb",
        "after_rgb",
        "after_semantic",
    ]
    assert evidence["row_count"] == 2
    assert evidence["callback_counts"] == [120, 120, 120]
    assert evidence["num_physics_steps"] == [122, 122, 122]
    assert evidence["simulation_time_s"] == [state["simulation_time_s"]] * 3
    assert evidence["particle_ids"] == [0, 1]
    assert evidence["proto_indices"] == [0, 0]
    for field in (
        "positions",
        "simulation_positions",
        "velocities",
        "ids",
        "proto_indices",
    ):
        hashes = evidence["hashes"][field]
        assert len(hashes) == 3
        assert len(set(hashes)) == 1


@pytest.mark.parametrize(
    ("state_label", "field", "replacement", "predicate"),
    [
        ("after_rgb", "callback_count", 121, "callback_count"),
        ("after_semantic", "num_physics_steps", 123, "num_physics_steps"),
        ("after_semantic", "simulation_time_s", 2.0, "simulation_time_s"),
        (
            "after_rgb",
            "positions",
            [[9.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "positions",
        ),
        (
            "after_semantic",
            "velocities",
            [[1.0, 0.0, -0.1], [0.0, 0.0, -0.2]],
            "velocities",
        ),
        ("after_semantic", "ids", [0, 2], "ids"),
    ],
)
def test_paired_capture_evidence_names_drift_after_either_pass(
    state_label, field, replacement, predicate
):
    state = _paired_state()
    states = {
        "before_rgb": deepcopy(state),
        "after_rgb": deepcopy(state),
        "after_semantic": deepcopy(state),
    }
    states[state_label][field] = replacement

    evidence = probe._paired_capture_evidence(states)

    assert evidence["passed"] is False
    assert predicate in evidence["failed_predicates"]


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

    close_exit_codes = []

    class FakeApp:
        def __init__(self, _config):
            pass

        def close(self, *, exit_code):
            assert (output / "receipt.json").is_file()
            close_exit_codes.append(exit_code)
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
    assert close_exit_codes == [2]
    assert receipt["runtime"]["requested_process_exit_code"] == 2


def test_main_requests_zero_exit_code_when_receipt_passes(tmp_path, monkeypatch):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    output = tmp_path / "probe-output"
    close_exit_codes = []

    class FakeApp:
        def __init__(self, _config):
            pass

        def close(self, *, exit_code):
            assert (output / "receipt.json").is_file()
            close_exit_codes.append(exit_code)

    fake_isaacsim = ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = FakeApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)

    def pass_probe(_args, _app, receipt):
        receipt["status"] = "pass"

    monkeypatch.setattr(probe, "run_probe", pass_probe)

    assert probe.main(["--stage", str(stage), "--out", str(output)]) == 0
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert close_exit_codes == [0]
    assert receipt["runtime"]["requested_process_exit_code"] == 0

"""CPU tests for fail-closed outer validation of the local PhysX probe."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "validate_physx_isosurface_emission_probe",
    ROOT / "cli/twin/probes" / "validate_physx_isosurface_emission_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _receipt() -> dict[str, object]:
    condition = "a12_multi_disabled_toggle"
    return {
        "status": "completed",
        "condition_requested": condition,
        "errors": [],
        "conditions": [
            {
                "name": condition,
                "completed": True,
                "substantive_isosurface_observed": True,
                "first_substantive_artifact": {
                    "path": "condition/step-021.json",
                    "sha256": "a" * 64,
                    "mesh_sha256": "b" * 64,
                },
                "snapshots": [
                    {
                        "particles": {
                            "finite_positions": True,
                            "count_matches": True,
                            "movement": {
                                "motion_observed": True,
                                "gravity_moved_down": True,
                                "maximum_displacement_m": 0.01,
                            },
                        }
                    }
                ],
            }
        ],
    }


def test_clean_completed_condition_passes_outer_validation():
    assert (
        validator.validate_outer_evidence(
            _receipt(),
            log_text="[4.7s] Simulation App Startup Complete\nclean shutdown\n",
            expected_condition="a12_multi_disabled_toggle",
        )
        == []
    )


def test_gpu_kernel_and_cuda_stop_log_errors_are_real_failures():
    failures = validator.validate_outer_evidence(
        _receipt(),
        log_text=(
            "[Error] [omni.physx.plugin] PhysX error: GPU kernel failed\n"
            "Cuda context manager error, simulation will be stopped\n"
        ),
        expected_condition="a12_multi_disabled_toggle",
    )

    assert failures == ["physx_gpu_or_cuda_runtime_error"]


def test_missing_particle_motion_fails_even_if_process_exited_zero():
    receipt = deepcopy(_receipt())
    receipt["conditions"][0]["snapshots"][-1]["particles"]["movement"].update(
        motion_observed=False,
        gravity_moved_down=False,
        maximum_displacement_m=0.0,
    )

    failures = validator.validate_outer_evidence(
        receipt,
        log_text="clean shutdown",
        expected_condition="a12_multi_disabled_toggle",
    )

    assert "particle_motion_not_observed" in failures


def test_a15_condition_requires_first_emission_creation_and_all_update_boundaries():
    receipt = _receipt()
    condition_name = "nvidia_demo_shared_create_first_emission_update_boundary"
    receipt["condition_requested"] = condition_name
    receipt["conditions"][0]["name"] = condition_name
    receipt["conditions"][0]["particle_set_created_step"] = 6
    receipt["conditions"][0]["pre_sim_app_update_count"] = 19

    failures = validator.validate_outer_evidence(
        receipt,
        log_text="clean shutdown",
        expected_condition=condition_name,
    )

    assert "demo_aligned_update_boundary_contract" in failures
    assert "substantive_changing_mesh_not_observed" in failures

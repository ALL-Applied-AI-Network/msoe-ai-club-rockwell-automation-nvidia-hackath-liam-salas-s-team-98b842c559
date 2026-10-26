from __future__ import annotations

from copy import deepcopy

import numpy as np

from fluidmix.parse_registration_probe import (
    A37_INPUT_ARRAYS,
    EXPECTED_PARTICLE_COUNT,
    PHYSICS_STEPS,
    STAGE2_LOWER_M,
    STAGE2_UPPER_M,
    build_probe_plan,
    evaluate_probe,
    particle_metrics,
    validate_input_arrays,
    validate_probe_receipt,
)


def _compact_rows() -> np.ndarray:
    rows = np.zeros((EXPECTED_PARTICLE_COUNT, 3), dtype=np.float64)
    rows[:, 0] = np.arange(EXPECTED_PARTICLE_COUNT) % 10 * 0.001
    rows[:, 1] = np.arange(EXPECTED_PARTICLE_COUNT) // 10 % 10 * 0.001
    rows[:, 2] = np.arange(EXPECTED_PARTICLE_COUNT) // 100 * 0.003
    return rows


def _input_arrays() -> dict[str, np.ndarray]:
    count = EXPECTED_PARTICLE_COUNT
    compositions = np.zeros((count, 3), dtype=np.float64)
    compositions[: count // 2, 0] = 1.0
    compositions[count // 2 :, 1] = 1.0
    return {
        "raw_simulation_positions": _compact_rows(),
        "velocities": np.zeros((count, 3), dtype=np.float64),
        "scales": np.ones((count, 3), dtype=np.float64),
        "ids": np.arange(count, dtype=np.int64),
        "proto_indices": np.zeros(count, dtype=np.int64),
        "compositions": compositions,
    }


def test_plan_freezes_saved_a37_state_and_parse_time_contrast() -> None:
    plan = build_probe_plan()

    assert plan["physics_steps"] == PHYSICS_STEPS
    assert plan["input"]["particle_count"] == EXPECTED_PARTICLE_COUNT
    assert plan["input"]["arrays"] == A37_INPUT_ARRAYS
    assert plan["intervention"]["particle_set_nonempty_at_helper_call"] is True
    assert plan["intervention"]["particle_set_nonempty_before_first_post_authoring_update"] is True
    assert plan["intervention"]["particle_set_nonempty_before_timeline_play"] is True
    assert plan["intervention"]["runtime_append_count"] == 0
    assert plan["decision"]["support_lower_inclusive_m"] == STAGE2_LOWER_M
    assert plan["decision"]["support_upper_inclusive_m"] == STAGE2_UPPER_M
    assert plan["decision"]["rescores_a34_or_a37"] is False
    assert plan["decision"]["physical_volume_gate_changed"] is False
    assert plan["claims"]["volume_stage_accepted"] is False
    assert plan["claims"]["render_or_rgb_created"] is False


def test_input_validator_requires_exact_shapes_ids_and_simplex() -> None:
    arrays = _input_arrays()
    assert validate_input_arrays(arrays) == ()

    broken = deepcopy(arrays)
    broken["ids"] = broken["ids"][::-1]
    assert "input_ids_mismatch" in validate_input_arrays(broken)

    broken = deepcopy(arrays)
    broken["compositions"][0] = (0.2, 0.2, 0.2)
    assert "input_compositions_mismatch" in validate_input_arrays(broken)

    broken = deepcopy(arrays)
    broken["velocities"][0, 0] = np.nan
    assert "input_nonfinite" in validate_input_arrays(broken)


def test_particle_metrics_detect_exact_overlap() -> None:
    rows = _compact_rows()
    clean = particle_metrics(rows)
    rows[1] = rows[0]
    overlap = particle_metrics(rows)

    assert clean["exact_duplicate_pair_count"] == 0
    assert overlap["exact_duplicate_pair_count"] == 1
    assert overlap["nearest_neighbor_min_m"] == 0.0


def test_evaluation_uses_frozen_stage2_band_without_rescoring_control() -> None:
    initial = {
        "wall_to_center_support_m": 0.029267833180104643,
        "nearest_neighbor_q50_m": 0.0014566168517088974,
        "exact_duplicate_pair_count": 10,
    }
    expanded = {
        "wall_to_center_support_m": STAGE2_LOWER_M,
        "nearest_neighbor_q50_m": 0.0028,
        "exact_duplicate_pair_count": 0,
    }
    unchanged = dict(initial)

    passed = evaluate_probe(initial, expanded)
    failed = evaluate_probe(initial, unchanged)

    assert passed["passed"] is True
    assert passed["decision"] == "supports_runtime_registration_cause"
    assert failed["passed"] is False
    assert failed["decision"] == "falsified_parse_time_registration_cause"
    assert "final_support_outside_frozen_stage2_band" in failed["failed_predicates"]
    assert passed["rescores_a34_or_a37"] is False


def _passing_receipt() -> dict[str, object]:
    plan = build_probe_plan()
    initial = {
        "wall_to_center_support_m": 0.029267833180104643,
        "nearest_neighbor_q50_m": 0.0014566168517088974,
        "exact_duplicate_pair_count": 10,
    }
    final = {
        "wall_to_center_support_m": STAGE2_LOWER_M,
        "nearest_neighbor_q50_m": 0.0028,
        "exact_duplicate_pair_count": 0,
    }
    return {
        "schema_version": plan["schema_version"],
        "experiment": plan["experiment"],
        "status": "pass",
        "plan": plan,
        "runtime": {
            "render_product_created": False,
            "rgb_captured": False,
            "video_created": False,
            "physics_callbacks": PHYSICS_STEPS,
            "callback_contract_passed": True,
        },
        "input": {
            "hashes_match_frozen": True,
            "arrays_valid": True,
        },
        "particle_contract": {
            "nonempty_at_helper_call": True,
            "nonempty_before_first_post_authoring_update": True,
            "nonempty_before_timeline_play": True,
            "runtime_append_count": 0,
            "fluid": True,
            "self_collision": True,
            "particle_group": 0,
            "particle_system_targets": ["/FillStation/FluidmixF2/ParticleSystem"],
            "mass_readback_matches": True,
            "density_readback_matches": True,
        },
        "snapshots": {
            "initial_authored": {
                "count": EXPECTED_PARTICLE_COUNT,
                "inside_count": EXPECTED_PARTICLE_COUNT,
                "outside_count": 0,
                "escaped_count": 0,
                "ids_exact": True,
                "finite": True,
                "metrics": initial,
            },
            "final_simulated": {
                "count": EXPECTED_PARTICLE_COUNT,
                "inside_count": EXPECTED_PARTICLE_COUNT,
                "outside_count": 0,
                "escaped_count": 0,
                "ids_exact": True,
                "finite": True,
                "metrics": final,
            },
        },
        "hypothesis_evaluation": evaluate_probe(initial, final),
        "experiment_outcome": "supports_runtime_registration_cause",
        "artifacts": {"manifest_valid": True, "required_arrays_valid": True},
        "errors": [],
    }


def test_receipt_validator_fails_closed_on_substantive_missing_or_false_evidence() -> None:
    receipt = _passing_receipt()
    assert validate_probe_receipt(receipt) == ()

    broken = deepcopy(receipt)
    broken["particle_contract"]["fluid"] = False
    assert "particle_contract_mismatch" in validate_probe_receipt(broken)

    broken = deepcopy(receipt)
    broken["snapshots"]["final_simulated"]["outside_count"] = 1
    assert "final_snapshot_mismatch" in validate_probe_receipt(broken)

    broken = deepcopy(receipt)
    broken["runtime"]["rgb_captured"] = True
    assert "unexpected_render_output" in validate_probe_receipt(broken)

    broken = deepcopy(receipt)
    broken["experiment_outcome"] = "supports_runtime_registration_cause"
    broken["hypothesis_evaluation"]["decision"] = (
        "falsified_parse_time_registration_cause"
    )
    assert "hypothesis_decision_mismatch" in validate_probe_receipt(broken)

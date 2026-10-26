from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fluidmix.explicit_mass_probe import (
    A36_ATTEMPT2_RECEIPT_SHA256,
    EXPERIMENT,
    PARTICLE_MASS_KG,
    STAGE1_LOWER_M,
    STAGE1_TARGET_M,
    STAGE1_UPPER_M,
    STAGE2_LOWER_M,
    STAGE2_TARGET_M,
    STAGE2_UPPER_M,
    build_probe_plan,
    evaluate_hypothesis,
    expected_total_mass_kg,
    frozen_mass_transition_schedule,
    nearest_neighbor_diagnostic,
    validate_mass_transitions,
)

SCRIPTS_ROOT = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
import physx_explicit_mass_probe as runtime_probe


def _transition(*, ordinal: int, count: int) -> dict[str, object]:
    total = expected_total_mass_kg(count)
    return {
        "transition_ordinal": ordinal,
        "physics_step": ordinal * 12,
        "active_particle_count": count,
        "per_particle_mass_kg": PARTICLE_MASS_KG,
        "requested_total_mass_kg": total,
        "readback_total_mass_kg": total,
        "readback_per_particle_mass_kg": total / count,
        "mass_attr_authored": True,
        "density_readback_kg_m3": 1000.0,
        "density_attr_authored": True,
        "mass_overrides_density": True,
        "set_succeeded": True,
        "readback_matches": True,
    }


def test_a37_plan_freezes_single_mass_intervention() -> None:
    plan = build_probe_plan()

    assert plan["experiment"] == EXPERIMENT
    assert plan["seed"] == 20261025
    assert plan["capture_steps"] == [1412, 2622]
    assert plan["render_product"] is False
    assert plan["rgb"] is False
    assert plan["video"] is False
    assert plan["denominator"] == 1
    assert plan["control"]["a36_attempt2_receipt_sha256"] == A36_ATTEMPT2_RECEIPT_SHA256
    assert plan["intervention"] == {
        "name": "explicit_runtime_total_mass",
        "formula": "8*fluid_rest_offset_m^3*density_kg_m3",
        "fluid_rest_offset_m": 0.0015,
        "density_kg_m3": 1000.0,
        "per_particle_mass_kg": pytest.approx(2.7e-5),
        "after_each_emission_append": True,
        "total_mass_formula": "active_particle_count*per_particle_mass_kg",
        "transition_count": 219,
        "readback_absolute_tolerance_kg": 1.0e-8,
    }
    assert plan["stage1_decision"] == {
        "metric": "raw_wall_to_center_support_m",
        "target_m": pytest.approx(0.01912223523598308),
        "tolerance_m": 0.006,
        "lower_inclusive_m": pytest.approx(0.01312223523598308),
        "upper_inclusive_m": pytest.approx(0.02512223523598308),
        "pass_if_inside_band_and_all_other_gates_pass": True,
        "falsified_if_below_lower_m": True,
    }
    assert plan["stage2_decision"] == {
        "metric": "raw_wall_to_center_support_m",
        "target_m": pytest.approx(0.03824447047196616),
        "tolerance_m": 0.006,
        "lower_inclusive_m": pytest.approx(0.03224447047196616),
        "upper_inclusive_m": pytest.approx(0.04424447047196616),
        "pass_if_inside_band_and_all_other_gates_pass": True,
        "falsified_if_below_lower_m": True,
    }
    assert plan["identical_to_a36_except"] == [
        "point_instancer_particle_mass_argument_kg",
        "runtime_total_mass_authored_after_each_emission_append",
        "mass_transition_receipt",
    ]
    assert plan["claims"]["fluid_rest_offset_is_render_radius"] is False
    assert plan["claims"]["rescores_a34"] is False


def test_documented_equivalent_mass_formula_and_stage_totals() -> None:
    assert PARTICLE_MASS_KG == pytest.approx(2.7e-5, rel=0.0, abs=1e-15)
    assert expected_total_mass_kg(4) == pytest.approx(0.000108)
    assert expected_total_mass_kg(370) == pytest.approx(0.00999)
    assert expected_total_mass_kg(740) == pytest.approx(0.01998)
    with pytest.raises((TypeError, ValueError)):
        expected_total_mass_kg(0)


def test_frozen_mass_transition_schedule_matches_first_two_a36_pours() -> None:
    schedule = frozen_mass_transition_schedule()

    assert len(schedule) == 219
    assert schedule[0] == {
        "transition_ordinal": 1,
        "physics_step": 12,
        "active_particle_count": 4,
    }
    assert schedule[-1] == {
        "transition_ordinal": 219,
        "physics_step": 2622,
        "active_particle_count": 740,
    }


def test_mass_transition_validation_accepts_exact_readbacks() -> None:
    records = [_transition(ordinal=1, count=4), _transition(ordinal=2, count=7)]

    assert validate_mass_transitions(records, expected_count=2) == ()


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("readback_total_mass_kg", 0.0, "mass_readback_mismatch"),
        ("mass_attr_authored", False, "mass_attr_not_authored"),
        ("density_attr_authored", False, "density_attr_not_authored"),
        ("density_readback_kg_m3", 999.0, "density_readback_mismatch"),
        ("active_particle_count", 0, "mass_transition_invalid"),
    ],
)
def test_mass_transition_validation_fails_closed(
    field: str, value: object, failure: str
) -> None:
    records = [_transition(ordinal=1, count=4)]
    records[0][field] = value

    assert failure in validate_mass_transitions(records, expected_count=1)


def test_runtime_mass_transition_authors_total_mass_and_reads_it_back() -> None:
    class Attribute:
        def __init__(self) -> None:
            self.value = 0.0
            self.authored = False

        def Set(self, value: float) -> bool:
            self.value = float(value)
            self.authored = True
            return True

        def Get(self) -> float:
            return self.value

        def HasAuthoredValueOpinion(self) -> bool:
            return self.authored

    class MassAPI:
        def __init__(self) -> None:
            self.attribute = Attribute()
            self.density = Attribute()
            self.density.Set(1000.0)

        def CreateMassAttr(self) -> Attribute:
            return self.attribute

        def GetDensityAttr(self) -> Attribute:
            return self.density

    record = runtime_probe._author_total_mass_transition(
        MassAPI(),
        transition_ordinal=3,
        physics_step=36,
        active_particle_count=11,
    )

    assert record["requested_total_mass_kg"] == pytest.approx(11 * 2.7e-5)
    assert record["readback_total_mass_kg"] == pytest.approx(11 * 2.7e-5)
    assert record["readback_per_particle_mass_kg"] == pytest.approx(2.7e-5)
    assert record["mass_attr_authored"] is True
    assert record["density_readback_kg_m3"] == 1000.0
    assert record["density_attr_authored"] is True
    assert record["readback_matches"] is True


def test_nearest_neighbor_diagnostic_is_fixed_and_duplicate_sensitive() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.003, 0.0, 0.0]],
        dtype=np.float64,
    )

    result = nearest_neighbor_diagnostic(positions)

    assert result["count"] == 3
    assert result["minimum_m"] == 0.0
    assert result["maximum_m"] == pytest.approx(0.003)
    assert result["fraction_below_0_0027"] == pytest.approx(2 / 3)
    assert result["fraction_above_0_0033"] == 0.0
    assert result["acceptance_gate"] is False


def _result_receipt(
    stage2_support_m: float, *, stage1_support_m: float = STAGE1_TARGET_M
) -> dict[str, object]:
    return {
        "status": "pass",
        "errors": [],
        "stages": [
            {
                "physics_step": 1412,
                "inside_vial_interior_count": 370,
                "outside_vial_interior_count": 0,
                "escaped_particle_count": 0,
                "ids_exact": True,
                "diagnostics": {
                    "raw_simulation_positions": {
                        "wall_to_center_support_m": stage1_support_m
                    }
                },
            },
            {
                "physics_step": 2622,
                "inside_vial_interior_count": 740,
                "outside_vial_interior_count": 0,
                "escaped_particle_count": 0,
                "ids_exact": True,
                "diagnostics": {
                    "raw_simulation_positions": {
                        "wall_to_center_support_m": stage2_support_m
                    }
                },
            },
        ],
    }


def test_stage2_hypothesis_evaluation_uses_frozen_band() -> None:
    passed = evaluate_hypothesis(_result_receipt(STAGE2_TARGET_M))
    below = evaluate_hypothesis(_result_receipt(STAGE2_LOWER_M - 1e-9))
    above = evaluate_hypothesis(_result_receipt(STAGE2_UPPER_M + 1e-9))
    stage1_below = evaluate_hypothesis(
        _result_receipt(
            STAGE2_TARGET_M, stage1_support_m=STAGE1_LOWER_M - 1e-9
        )
    )
    stage1_above = evaluate_hypothesis(
        _result_receipt(
            STAGE2_TARGET_M, stage1_support_m=STAGE1_UPPER_M + 1e-9
        )
    )
    leakage_receipt = _result_receipt(STAGE2_TARGET_M)
    leakage_receipt["stages"][1]["escaped_particle_count"] = 1
    leakage = evaluate_hypothesis(leakage_receipt)

    assert passed["decision"] == "pass"
    assert below["decision"] == "falsified_stage2_below_band"
    assert above["decision"] == "fail_stage2_above_band"
    assert stage1_below["decision"] == "falsified_stage1_below_band"
    assert stage1_above["decision"] == "fail_stage1_above_band"
    assert leakage["decision"] == "failed_other_gate"
    assert passed["rescores_a34"] is False


def test_plan_only_writes_distinct_a37_plan_without_isaac(tmp_path: Path) -> None:
    script = SCRIPTS_ROOT / "physx_explicit_mass_probe.py"
    output = tmp_path / "plan"
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--out",
            str(output),
            "--stage",
            str(stage.resolve()),
            "--plan-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((output / "plan.json").read_text(encoding="utf-8")) == build_probe_plan()


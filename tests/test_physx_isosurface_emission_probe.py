"""CPU contracts for the bounded F2 generated-isosurface emission ablation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "physx_isosurface_emission_probe",
    ROOT / "scripts" / "physx_isosurface_emission_probe.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_frozen_probe_plan_separates_emission_and_visibility_questions():
    plan = probe.build_probe_plan(seed=20261025)

    assert plan["schema_version"] == 1
    assert plan["seed"] == 20261025
    assert plan["fixed_dt_s"] == 1.0 / 60.0
    assert plan["emission_steps"] == list(range(6, 121, 6))
    assert plan["particles_per_emission"] == 4
    assert plan["snapshot_steps"] == [0, 1, 6, 12, 24, 60, 66, 120, 126]
    assert plan["conditions"] == [
        {
            "name": "a12_multi_disabled_toggle",
            "particle_set_model": "twenty_preauthored_disabled_sets_then_enable",
            "source_visibility": "inherited_for_entire_condition",
            "max_step": 120,
        },
        {
            "name": "nvidia_demo_shared_append_visibility_ablation",
            "particle_set_model": "one_enabled_points_set_runtime_array_append",
            "source_visibility": "inherited_through_step_60_hidden_61_120_restored_121_126",
            "max_step": 126,
        },
    ]
    assert plan["hypotheses"] == {
        "emission_model": (
            "the demo-style shared runtime-append set yields substantive generated "
            "isosurface geometry while A12-style disabled-set toggles do not"
        ),
        "visibility": (
            "after a substantive shared-set surface exists, hiding the generated "
            "source prevents later USD mesh updates"
        ),
    }
    assert len(plan["falsifiers"]) == 2


def test_parser_requires_output_and_accepts_only_frozen_seed():
    args = probe.build_parser().parse_args(
        ["--out", "probe-output", "--seed", "20261025"]
    )

    assert args.out == Path("probe-output")
    assert args.seed == 20261025


def test_script_has_no_renderer_or_replicator_capture_path():
    source = (ROOT / "scripts" / "physx_isosurface_emission_probe.py").read_text(
        encoding="utf-8"
    )

    assert "omni.replicator" not in source
    assert "render_product" not in source
    assert "PathTracing" not in source

"""CPU boundaries for the dedicated render-free Q2 entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence", ROOT / "scripts" / "physx_three_pour_sequence.py"
)
assert SPEC is not None and SPEC.loader is not None
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)


def _args(tmp_path: Path, *extra: str):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    return sequence.build_parser().parse_args(
        ["--out", str(tmp_path / "out"), "--stage", str(stage), *extra]
    )


def test_q2_parser_receipt_is_explicitly_render_free(tmp_path: Path) -> None:
    args = _args(tmp_path, "--q2-no-render")
    sequence._validate_args(args)
    receipt = sequence._empty_q2_no_render_receipt(args)

    assert args.q2_no_render is True
    assert receipt["artifact_type"] == "q2_no_render_qualification"
    assert receipt["target"]["physics_step"] == 1412
    assert receipt["target"]["expected_particle_count"] == 640
    assert receipt["target"]["stream_hidden"] is True
    assert receipt["target"]["rendering_disabled"] is True
    assert receipt["render"] == {
        "enabled": False,
        "capture_enabled": False,
        "render_product_created": False,
        "rgb_artifacts": [],
        "semantic_artifacts": [],
        "video_artifacts": [],
    }
    assert receipt["runtime"]["max_particles"] == 4096
    assert receipt["evidence"]["mass"]["expected_total_mass_kg"] == 0.01


def test_q2_ablation_records_settle_horizon_and_solver_iterations(
    tmp_path: Path,
) -> None:
    args = _args(
        tmp_path,
        "--q2-no-render",
        "--q2-post-emission-settle-steps",
        "360",
        "--q2-solver-position-iterations",
        "8",
    )
    sequence._validate_args(args)
    receipt = sequence._empty_q2_no_render_receipt(args)

    assert receipt["target"] == {
        **receipt["target"],
        "emission_end_step": 1412,
        "post_emission_settle_steps": 360,
        "post_emission_settle_s": 3.0,
        "physics_step": 1772,
    }
    assert receipt["runtime"]["solver_position_iterations_requested"] == 8
    assert receipt["ablation"] == {
        "diagnostic_only": True,
        "production_defaults_modified": False,
        "emission_end_step": 1412,
        "analysis_step": 1772,
        "post_emission_settle_steps": 360,
        "post_emission_settle_s": 3.0,
        "solver_position_iterations": 8,
    }


def test_q2_settle_suppresses_every_later_pour_event(tmp_path: Path) -> None:
    args = _args(
        tmp_path,
        "--q2-no-render",
        "--q2-post-emission-settle-steps",
        "360",
    )
    sequence._validate_args(args)
    plan = sequence._empty_q2_no_render_receipt(args)["plan"]

    indexed = sequence._q2_first_pour_event_index(plan)
    events = [event for rows in indexed.values() for event in rows]

    assert indexed
    assert max(indexed) <= 1412
    assert {event["pour_ordinal"] for event in events} == {1}
    assert sum(event["particle_count"] for event in events) == 640


def test_q2_diffusion_degree_is_not_a_native_neighbor_gate() -> None:
    diagnostic = sequence._q2_diffusion_neighbor_diagnostic(
        max_degree=121,
        physx_max_neighborhood=96,
    )

    assert diagnostic == {
        "physx_max_neighborhood": 96,
        "max_observed_diffusion_degree": 121,
        "cap_saturated": None,
        "unsaturated": None,
        "measurement_source": "runtime_diffusion_graph_not_physx_neighbor_readback",
        "gate_required": False,
        "passed": None,
        "status": "diagnostic_only",
    }


@pytest.mark.parametrize(
    "extra",
    [
        ("--q2-post-emission-settle-steps", "1"),
        ("--q2-solver-position-iterations", "8"),
    ],
)
def test_q2_ablation_flags_require_q2_mode(
    tmp_path: Path, extra: tuple[str, ...]
) -> None:
    args = _args(tmp_path, *extra)
    with pytest.raises(ValueError, match="Q2 ablation"):
        sequence._validate_args(args)


@pytest.mark.parametrize(
    "extra",
    [
        ("--q2-no-render", "--qualification-mode"),
        ("--q2-no-render", "--count-driven-twin-liquid-mesh"),
    ],
)
def test_q2_rejects_render_or_full_run_modes(tmp_path: Path, extra: tuple[str, ...]) -> None:
    args = _args(tmp_path, *extra)
    with pytest.raises(ValueError, match="q2_no_render"):
        sequence._validate_args(args)

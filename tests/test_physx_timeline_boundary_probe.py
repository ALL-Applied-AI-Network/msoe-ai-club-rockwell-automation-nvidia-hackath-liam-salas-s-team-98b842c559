"""Pure contracts for the bounded Isaac timeline-boundary probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "physx_timeline_boundary_probe.py"
SPEC = importlib.util.spec_from_file_location("physx_timeline_boundary_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _callbacks(count: int = 2750) -> list[dict[str, object]]:
    return [
        {
            "callback_index": index,
            "dt_s": 1.0 / 120.0,
            "timeline_time_s": index / 120.0,
            "app_update_index": index,
        }
        for index in range(1, count + 1)
    ]


def test_boundary_probe_accepts_monotonic_120hz_trace_past_source_stage_end():
    evidence = probe.validate_callback_trace(
        _callbacks(), physics_hz=120, target_callback_count=2750
    )

    assert evidence["passed"] is True
    assert evidence["failed_predicates"] == []
    assert evidence["callback_count"] == 2750
    assert evidence["source_stage_end_time_s"] == 686.0 / 30.0
    assert evidence["crossed_source_stage_end"] is True
    assert [row["callback_index"] for row in evidence["boundary_window"]] == list(
        range(2743, 2751)
    )
    assert evidence["boundary_window"][3]["timeline_time_s"] == 2746.0 / 120.0


def test_boundary_probe_rejects_the_former_wrap_signature():
    callbacks = _callbacks()
    callbacks[2745]["timeline_time_s"] = 1.0 / 120.0

    evidence = probe.validate_callback_trace(
        callbacks, physics_hz=120, target_callback_count=2750
    )

    assert evidence["passed"] is False
    assert "timeline_time_not_strictly_monotonic" in evidence["failed_predicates"]
    assert "timeline_time_not_callback_index_over_hz" in evidence["failed_predicates"]

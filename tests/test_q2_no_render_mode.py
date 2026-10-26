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

"""Pure contracts for the non-particle-colour F2 fallback runner."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "physx_three_pour_sequence_fallback",
    ROOT / "scripts" / "physx_three_pour_sequence.py",
)
assert SPEC is not None and SPEC.loader is not None
sequence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sequence)


def test_runtime_mutations_execute_in_the_stronger_session_edit_target():
    events: list[object] = []

    class Stage:
        edit_target = "root"

    class Context:
        def __init__(self, stage, layer):
            self.stage = stage
            self.layer = layer
            self.previous = stage.edit_target

        def __enter__(self):
            self.stage.edit_target = self.layer
            events.append(("enter", self.layer))

        def __exit__(self, *_):
            events.append(("exit", self.layer))
            self.stage.edit_target = self.previous

    class Usd:
        EditContext = Context

    stage = Stage()

    observed = sequence._with_session_edit_context(
        Usd,
        stage,
        "session",
        lambda: (events.append(("action", stage.edit_target)), stage.edit_target)[1],
    )

    assert observed == "session"
    assert stage.edit_target == "root"
    assert events == [
        ("enter", "session"),
        ("action", "session"),
        ("exit", "session"),
    ]
    source = inspect.getsource(sequence.run_sequence)
    assert source.count("_with_session_edit_context(") >= 3


def test_semantic_annotator_is_attached_after_runtime_labels_and_stage_attach():
    source = inspect.getsource(sequence.run_sequence)

    label_offset = source.index("system_semantics = UsdSemantics.LabelsAPI.Apply")
    stage_attach_offset = source.index("simulation.attach_stage(stage_id)")
    semantic_attach_offset = source.index(
        'semantic_annotator = rep.AnnotatorRegistry.get_annotator(\n'
        '            "semantic_segmentation"'
    )

    assert label_offset < stage_attach_offset < semantic_attach_offset


def test_event_index_covers_exact_frozen_back_to_back_schedule():
    plan = sequence.build_three_pour_plan()

    events = sequence._build_emission_event_index(plan)

    flattened = [event for step in sorted(events) for event in events[step]]
    assert sum(event["particle_count"] for event in flattened) == 2221
    assert [event["tank_id"] for event in flattened[:2]] == ["t200", "t200"]
    assert flattened[-1]["tank_id"] == "t100"
    assert min(events) > 0
    assert max(events) == 1984


@pytest.mark.parametrize(
    ("step", "tank"),
    [(1, "t200"), (706, "t200"), (707, "t400"), (1311, "t400"), (1312, "t100"), (1984, "t100"), (1985, None), (2002, None)],
)
def test_render_only_stream_uses_source_tank_for_every_pour_step(step, tank):
    assert sequence._active_tank_for_step(sequence.build_three_pour_plan(), step) == tank


def test_global_blend_updates_during_emission_by_actual_particle_count():
    counts = np.zeros(3, dtype=np.float64)

    first = sequence._update_global_component_counts(counts, "t200", 10)
    second = sequence._update_global_component_counts(first, "t400", 10)
    final = sequence._update_global_component_counts(second, "t100", 40)

    np.testing.assert_array_equal(counts, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(first / first.sum(), [1.0, 0.0, 0.0])
    np.testing.assert_allclose(second / second.sum(), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(final / final.sum(), [1 / 6, 1 / 6, 4 / 6])
    np.testing.assert_allclose(
        sequence._global_transmission_color(final),
        np.asarray([0.95, 0.38, 0.08]) / 6
        + np.asarray([0.18, 0.55, 0.75]) / 6
        + np.asarray([0.92, 0.98, 1.0]) * 4 / 6,
    )


def test_global_blend_material_update_returns_live_usd_readback():
    class Input:
        value = None

        def Set(self, value):
            self.value = value
            return True

        def Get(self):
            return self.value

    class Shader:
        input = Input()

        def GetInput(self, name):
            assert name == "specular_transmission_color"
            return self.input

    class Gf:
        @staticmethod
        def Vec3f(*values):
            return tuple(float(value) for value in values)

    assert sequence._set_global_blend_color(Shader(), Gf, [0.1, 0.2, 0.3]) == (
        0.1,
        0.2,
        0.3,
    )


@pytest.mark.parametrize(("tank", "count"), [("bad", 1), ("t200", 0), ("t200", True)])
def test_global_blend_rejects_invalid_emission(tank, count):
    with pytest.raises((TypeError, ValueError)):
        sequence._update_global_component_counts(np.zeros(3), tank, count)


def test_capture_steps_add_exact_stage_frames_without_settling():
    steps = sequence._build_capture_steps(sequence.build_three_pour_plan(), stride=6)

    assert steps == tuple(sorted(set(steps)))
    assert steps[0] == 6
    assert steps[-1] == 2002
    assert len(steps) == 336
    assert {706, 1311, 2002}.issubset(steps)


@pytest.mark.parametrize(
    "extra",
    [
        ["--capture-stride", "7"],
        ["--video-fps", "9"],
    ],
)
def test_runner_rejects_noncontract_capture_timeline(tmp_path, extra):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", str(tmp_path / "out"), "--stage", str(stage), *extra]
    )

    with pytest.raises(ValueError, match="10 Hz|capture_stride"):
        sequence._validate_args(args)


def test_receipt_predeclares_honest_fallback_claims(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", str(tmp_path / "out"), "--stage", str(stage)]
    )

    receipt = sequence._empty_receipt(args)

    plan = sequence.build_three_pour_plan()
    assert receipt["schema_version"] == 4
    assert receipt["fallback_evidence"] == plan["fallback_trigger"]
    assert receipt["material"]["vial_representation"] == "visible_isosurface_one_global_material"
    assert receipt["material"]["stream_representation"] == "render_only_mesh_current_tank_material"
    assert receipt["stream_render"]["records"] == []
    assert receipt["global_blend"]["model"] == "pbd_motion_global_vial_blend_render_stream_fallback_v1"
    assert receipt["global_blend"]["blend_update_timing"] == "every_emission_event"
    assert receipt["global_blend"]["particle_composition_tracked"] is False
    assert receipt["global_blend"]["diffusion_simulated"] is False
    assert receipt["global_blend"]["spatial_mixing_color_rendered"] is False
    assert receipt["semantics"]["semantic_label"] == "liquid"
    assert receipt["semantics"]["semantic_type"] == "class"
    assert receipt["semantics"]["training_eligible"] is False
    assert (
        receipt["semantics"]["ineligibility_reason"]
        == "qualification_run_not_case_dataset"
    )
    assert receipt["claims"]["particle_composition_tracked"] is False
    assert receipt["claims"]["dye_concentration_simulated"] is False
    assert receipt["claims"]["sim_to_real_validated"] is False
    assert receipt["claims"] == plan["claims"]
    assert receipt["mixing_assumption"] == plan["mixing_assumption"]
    assert receipt["timing"]["no_inter_pour_settle"] is True


def test_runner_source_has_no_obsolete_settling_or_particle_colour_claim():
    source = (ROOT / "scripts" / "physx_three_pour_sequence.py").read_text(encoding="utf-8")

    assert "stable_min_steps" not in source
    assert "stable_max_steps" not in source
    assert "PointInstancer" not in source
    assert "diffuse_particle_compositions" not in source
    assert "fully-mixed" not in source

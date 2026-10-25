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


def test_gb10_uma_memory_falls_back_to_system_wide_meminfo(monkeypatch):
    receipt = {
        "memory": {
            "samples_mib": [],
            "peak_used_mib": None,
            "hard_limit_exceeded": False,
            "sample_source": None,
            "sample_scope": None,
        },
        "warnings": [],
    }
    monkeypatch.setattr(sequence, "_gpu_used_memory_mib", lambda: None)
    monkeypatch.setattr(sequence, "_gb10_uma_memory_expected", lambda: True)
    monkeypatch.setattr(sequence, "_system_used_memory_mib", lambda: 6144)

    assert sequence._memory_sample(receipt) == 6144
    assert receipt["memory"]["samples_mib"] == [6144]
    assert receipt["memory"]["peak_used_mib"] == 6144
    assert receipt["memory"]["sample_source"] == "proc_meminfo_system_used_uma"
    assert receipt["memory"]["sample_scope"] == "system_wide_unified_memory"
    assert len(receipt["warnings"]) == 1


def test_missing_discrete_gpu_memory_sample_remains_a_failure(monkeypatch):
    receipt = {
        "memory": {
            "samples_mib": [],
            "peak_used_mib": None,
            "hard_limit_exceeded": False,
            "sample_source": None,
            "sample_scope": None,
        },
        "warnings": [],
    }
    monkeypatch.setattr(sequence, "_gpu_used_memory_mib", lambda: None)
    monkeypatch.setattr(sequence, "_gb10_uma_memory_expected", lambda: False)

    with pytest.raises(RuntimeError, match="GPU memory sample"):
        sequence._memory_sample(receipt)


def test_meminfo_parser_uses_total_minus_available():
    assert sequence._parse_meminfo_used_mib(
        "MemTotal: 131072 kB\nMemAvailable: 98304 kB\n"
    ) == 32

    with pytest.raises(ValueError, match="MemAvailable"):
        sequence._parse_meminfo_used_mib("MemTotal: 131072 kB\n")


def test_video_writer_uses_mjpg_avi_when_local_mp4v_is_unavailable(tmp_path):
    calls: list[tuple[str, str]] = []

    class Writer:
        def __init__(self, path, code):
            self.path = path
            self.code = code

        def isOpened(self):
            return self.code == "MJPG"

        def release(self):
            calls.append(("release", self.code))

    class Cv2:
        @staticmethod
        def VideoWriter_fourcc(*characters):
            return "".join(characters)

        @staticmethod
        def VideoWriter(path, code, fps, resolution):
            assert fps == 10.0
            assert resolution == (640, 480)
            calls.append((path, code))
            return Writer(path, code)

    writer, path, codec, fallback = sequence._open_video_writer(
        Cv2, tmp_path, width=640, height=480, fps=10.0
    )

    assert writer.isOpened()
    assert path == tmp_path / "three-pour.avi"
    assert codec == "MJPG"
    assert fallback is True
    assert calls[0] == (str(tmp_path / "three-pour.mp4"), "mp4v")
    assert calls[1] == ("release", "mp4v")
    assert calls[2] == (str(tmp_path / "three-pour.avi"), "MJPG")


def test_video_writer_fails_closed_when_no_codec_opens(tmp_path):
    class Writer:
        def isOpened(self):
            return False

        def release(self):
            pass

    class Cv2:
        VideoWriter_fourcc = staticmethod(lambda *characters: "".join(characters))
        VideoWriter = staticmethod(lambda *args: Writer())

    with pytest.raises(RuntimeError, match="mp4v or MJPG"):
        sequence._open_video_writer(
            Cv2, tmp_path, width=640, height=480, fps=10.0
        )


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


def test_runner_uses_the_a18_timeline_route_and_one_runtime_points_set():
    """F2 must not mix direct PhysX stepping with the proven timeline route."""

    source = inspect.getsource(sequence.run_sequence)

    assert "attach_stage" not in source
    assert ".simulate(" not in source
    assert "fetch_results" not in source
    assert "subscribe_physics_step_events" in source
    assert "timeline.set_time_codes_per_second(float(PHYSICS_HZ))" in source
    assert "timeline.set_ticks_per_frame(1)" in source
    assert "timeline.set_target_framerate(float(PHYSICS_HZ))" in source
    assert "timeline.set_play_every_frame(True)" in source
    assert "timeline.play()" in source
    assert "timeline.commit()" in source
    assert "shared_particle_points" in source
    assert "CreateIdsAttr" in source
    assert "GetIdsAttr" in source
    assert "runtime_particle_ids_exact_each_callback" in source
    assert "SETTING_UPDATE_VELOCITIES_TO_USD" in source
    assert "velocity_writeback_initial" in source
    assert "velocity_writeback_restored_readback" in source
    assert "active_prims" not in source
    assert "chunk_records_by_step" not in source
    assert "_replicator_same_state_step_checked" in source


def test_same_state_capture_rejects_a_callback_that_advances_physics():
    callbacks = [1]

    class Orchestrator:
        def step(self, **_kwargs):
            callbacks.append(2)

    rep = type("Rep", (), {"orchestrator": Orchestrator()})()
    timeline = type(
        "Timeline", (), {"get_current_time": lambda self: 0.0, "is_playing": lambda self: True}
    )()

    with pytest.raises(RuntimeError, match="advanced physics"):
        sequence._replicator_same_state_step_checked(
            rep,
            rt_subframes=8,
            callback_count=lambda: len(callbacks),
            timeline=timeline,
        )


def test_same_state_capture_rejects_a_playback_state_change():
    class Orchestrator:
        def step(self, **_kwargs):
            timeline.playing = True

    class Timeline:
        playing = False

        def get_current_time(self):
            return 0.0

        def is_playing(self):
            return self.playing

    timeline = Timeline()
    rep = type("Rep", (), {"orchestrator": Orchestrator()})()

    with pytest.raises(RuntimeError, match="playback"):
        sequence._replicator_same_state_step_checked(
            rep,
            rt_subframes=8,
            callback_count=lambda: 0,
            timeline=timeline,
        )


def test_semantic_annotator_is_attached_after_runtime_labels_on_timeline_route():
    source = inspect.getsource(sequence.run_sequence)

    label_offset = source.index("system_semantics = UsdSemantics.LabelsAPI.Apply")
    proxy_label_offset = source.index(
        "_author_direct_liquid_semantics(\n"
        "                    stage.GetPrimAtPath(LIQUID_PROXY_PATH)"
    )
    semantic_attach_offset = source.index(
        "semantic_annotator = rep.AnnotatorRegistry.get_annotator("
    )

    assert "attach_stage" not in source
    assert label_offset < proxy_label_offset < semantic_attach_offset
    assert "subscribe_physics_step_events" in source


def test_volume_capture_is_opted_in_and_read_back_before_semantic_annotator(tmp_path):
    class Settings:
        def __init__(self):
            self.values = {sequence.SYNTHETICDATA_CAPTURE_VOLUMES_SETTING: False}

        def get_as_bool(self, path):
            return bool(self.values.get(path, False))

        def set_bool(self, path, value):
            self.values[path] = bool(value)

    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    receipt = sequence._empty_receipt(
        sequence.build_parser().parse_args(
            ["--out", str(tmp_path / "out"), "--stage", str(stage)]
        )
    )
    settings = Settings()

    sequence._enable_syntheticdata_volume_capture(settings, receipt)

    assert receipt["semantics"]["capture_volumes_setting_path"] == (
        "/syntheticdata/sensors/captureVolumes"
    )
    assert receipt["semantics"]["capture_volumes_initial_readback"] is False
    assert receipt["semantics"]["capture_volumes_enabled_readback"] is True
    source = inspect.getsource(sequence.run_sequence)
    enable_offset = source.index("_enable_syntheticdata_volume_capture(")
    attach_offset = source.index(
        "semantic_annotator = rep.AnnotatorRegistry.get_annotator("
    )
    assert enable_offset < attach_offset


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
    assert receipt["schema_version"] == 12
    assert receipt["fallback_evidence"] == plan["fallback_trigger"]
    assert receipt["material"]["vial_representation"] == "exact_standard_mesh_proxy_one_global_material"
    assert receipt["material"]["stream_representation"] == "render_only_mesh_current_tank_material"
    assert receipt["stream_render"]["records"] == []
    assert receipt["global_blend"]["model"] == "timeline_owned_runtime_shared_points_stable_usd_ids_pbd_motion_composition_diffusion_global_blend_render_stream_fallback_v4"
    assert receipt["global_blend"]["blend_update_timing"] == "every_emission_event"
    assert receipt["global_blend"]["particle_composition_tracked"] is True
    assert receipt["global_blend"]["diffusion_simulated"] is True
    assert receipt["global_blend"]["spatial_mixing_color_rendered"] is False
    assert receipt["semantics"]["semantic_label"] == "liquid"
    assert receipt["semantics"]["semantic_type"] == "class"
    assert receipt["semantics"]["annotator"] == "semantic_segmentation"
    assert receipt["semantics"]["annotator_init_params"] == {
        "colorize": False,
        "semanticTypes": ["class"],
    }
    assert (
        receipt["semantics"]["mask_source"]
        == "exact_renderer_semantic_class_ids"
    )
    assert receipt["semantics"]["training_eligible"] is False
    assert (
        receipt["semantics"]["ineligibility_reason"]
        == "qualification_run_not_case_dataset"
    )
    assert receipt["claims"]["particle_composition_tracked"] is True
    assert receipt["claims"]["dye_concentration_simulated"] is False
    assert receipt["claims"]["sim_to_real_validated"] is False
    assert receipt["claims"] == plan["claims"]
    assert receipt["mixing_assumption"] == plan["mixing_assumption"]
    assert receipt["timing"]["no_inter_pour_settle"] is True
    assert receipt["semantics"]["capture_volumes_setting_path"] == (
        "/syntheticdata/sensors/captureVolumes"
    )
    assert receipt["semantics"]["capture_volumes_enabled_readback"] is False
    assert receipt["mesh_proxy"]["created"] is False
    assert receipt["mesh_proxy"]["source_path"].endswith("/Isosurface")
    assert receipt["mesh_proxy"]["proxy_path"].endswith(
        "/LiquidSemanticProxy"
    )
    assert receipt["mesh_proxy"]["refresh_records"] == []


def test_receipt_predeclares_ccd_and_internal_composition_evidence_contract(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", str(tmp_path / "out"), "--stage", str(stage)]
    )

    receipt = sequence._empty_receipt(args)

    assert receipt["schema_version"] == 12
    assert receipt["particle_system"]["ccd_enabled"] is False
    assert receipt["semantics"]["material_binding_authored"] is False
    assert receipt["semantics"]["material_binding"] is None
    assert receipt["semantics"]["isosurface_direct_binding_authored"] is False
    assert receipt["semantics"]["isosurface_direct_binding"] is None
    assert receipt["global_blend"]["particle_composition_tracked"] is True
    assert receipt["global_blend"]["diffusion_simulated"] is True
    evidence = receipt["composition_evidence"]
    assert evidence["internal_unrendered"] is True
    assert evidence["stages"] == []


def test_active_particle_rows_preserve_id_order_and_fail_closed_on_mismatch():
    records = [
        {"particle_ids": [10, 11], "particle_count": 2},
        {"particle_ids": [12], "particle_count": 1},
    ]
    compositions = {
        10: np.asarray([1.0, 0.0, 0.0]),
        11: np.asarray([0.0, 1.0, 0.0]),
        12: np.asarray([0.0, 0.0, 1.0]),
    }
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [0.004, 0.0, 0.0]],
        dtype=np.float64,
    )

    updated, diagnostics = sequence._diffuse_active_particle_rows(
        records, compositions, positions, [10, 11, 12]
    )

    assert diagnostics["particle_ids_sha256"]
    assert diagnostics["particle_row_order_preserved"] is True
    assert diagnostics["component_conservation_passed"] is True
    assert updated.shape == (3, 3)
    assert np.all(np.isfinite(updated))
    np.testing.assert_allclose(updated.sum(axis=0), [1.0, 1.0, 1.0])

    with pytest.raises(RuntimeError, match="count/order"):
        sequence._diffuse_active_particle_rows(
            records, compositions, positions[:2], [10, 11, 12]
        )

    with pytest.raises(RuntimeError, match="observed USD ids"):
        sequence._diffuse_active_particle_rows(
            records, compositions, positions, [11, 10, 12]
        )


def test_active_particle_diffusion_is_a_noop_before_first_emission():
    updated, diagnostics = sequence._diffuse_active_particle_rows(
        [], {}, np.empty((0, 3), dtype=np.float64), []
    )

    assert updated.shape == (0, 3)
    assert diagnostics["particle_count"] == 0
    assert diagnostics["neighbor_count"] == 0


def test_active_particle_diffusion_accepts_f2_scale_summation_roundoff():
    particle_count = 2221
    side = 14
    records = [{"particle_ids": list(range(particle_count)), "particle_count": particle_count}]
    compositions = {}
    for particle_id in range(particle_count):
        component = 0 if particle_id < 370 else 1 if particle_id < 740 else 2
        row = np.zeros(3, dtype=np.float64)
        row[component] = 1.0
        compositions[particle_id] = row
    positions = np.asarray(
        [
            (
                (index % side) * 0.003,
                ((index // side) % side) * 0.003,
                (index // (side * side)) * 0.003,
            )
            for index in range(particle_count)
        ],
        dtype=np.float64,
    )

    updated, diagnostics = sequence._diffuse_active_particle_rows(
        records, compositions, positions, list(range(particle_count))
    )

    assert updated.shape == (particle_count, 3)
    assert diagnostics["component_conservation_passed"] is True


def test_generated_isosurface_gets_a_direct_liquid_schema():
    class FakeAttribute:
        def __init__(self):
            self.value = None

        def Set(self, value):
            self.value = list(value)
            return True

        def Get(self):
            return self.value

    class FakePrim:
        def __init__(self):
            self.schemas = []

        def IsValid(self):
            return True

        def GetPath(self):
            return "/FillStation/FluidmixF2/ParticleSystem/Isosurface"

        def GetAppliedSchemas(self):
            return self.schemas

    class FakeLabels:
        def __init__(self, prim):
            self.prim = prim
            self.attribute = FakeAttribute()

        def __bool__(self):
            return True

        def CreateLabelsAttr(self):
            return self.attribute

        def GetLabelsAttr(self):
            return self.attribute

    class FakeLabelsAPI:
        @staticmethod
        def Apply(prim, taxonomy):
            prim.schemas.append(f"SemanticsLabelsAPI:{taxonomy}")
            return FakeLabels(prim)

    class FakeUsdSemantics:
        LabelsAPI = FakeLabelsAPI

    prim = FakePrim()

    evidence = sequence._author_direct_liquid_semantics(prim, FakeUsdSemantics)

    assert evidence == {
        "path": "/FillStation/FluidmixF2/ParticleSystem/Isosurface",
        "schema": "SemanticsLabelsAPI:class",
        "labels": ["liquid"],
        "direct_binding_verified": True,
    }


def test_runner_authors_material_semantics_before_annotator_attachment():
    source = (ROOT / "scripts" / "physx_three_pour_sequence.py").read_text(
        encoding="utf-8"
    )

    material_binding = source.index('"material_binding_authored": True')
    annotator_attachment = source.index(
        'semantic_annotator = rep.AnnotatorRegistry.get_annotator('
    )
    assert material_binding < annotator_attachment


def test_runner_source_has_no_obsolete_settling_or_particle_colour_claim():
    source = (ROOT / "scripts" / "physx_three_pour_sequence.py").read_text(encoding="utf-8")

    assert "stable_min_steps" not in source
    assert "stable_max_steps" not in source
    assert "PointInstancer" not in source
    assert "_diffuse_active_particle_rows" in source
    assert "fully-mixed" not in source

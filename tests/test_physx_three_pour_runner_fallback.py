"""Pure contracts for the non-particle-colour F2 fallback runner."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import types

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
    assert source.count("_with_session_edit_context(") >= 2


def test_runner_uses_the_a18_timeline_route_and_one_runtime_points_set():
    """F2 must not mix direct PhysX stepping with the proven timeline route."""

    source = inspect.getsource(sequence.run_sequence)

    assert "attach_stage" not in source
    assert ".simulate(" not in source
    assert "fetch_results" not in source
    assert "subscribe_physics_step_events" in source
    assert "_configure_zero_origin_timeline(" in source
    timeline_setup = inspect.getsource(sequence._configure_zero_origin_timeline)
    assert "timeline.set_time_codes_per_second(float(physics_hz))" in timeline_setup
    assert timeline_setup.index("timeline.set_start_time(0.0)") < timeline_setup.index(
        "timeline.set_current_time(0.0)"
    )
    assert "timeline.set_ticks_per_frame(1)" in timeline_setup
    assert "timeline.set_target_framerate(float(physics_hz))" in timeline_setup
    assert "timeline.set_play_every_frame(True)" in timeline_setup
    timeline_start = inspect.getsource(sequence._start_timeline_non_looping)
    assert "timeline.set_looping(False)" in timeline_start
    assert "timeline.play(looping=False)" in timeline_start
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
    assert source.count("_replicator_same_state_step_checked_in_session(") == 4
    assert "_replicator_same_state_step_checked(" not in source
    assert "PhysxParticleSetAPI.Apply" in source
    assert "PhysxParticleAPI(point_set_api)" in source
    assert "UsdPhysics.MassAPI.Apply" not in source
    assert "GetSimulationPointsAttr" in source
    assert "copy_exact_mesh_proxy(" not in source
    assert "stage_isosurface_same_state_sync" not in source


def test_motion_falsifier_precedes_same_step_capture_branch() -> None:
    source = inspect.getsource(sequence.run_sequence)

    assert sequence.MOTION_FALSIFIER_STEP == 12
    assert source.index("if global_step == MOTION_FALSIFIER_STEP:") < source.index(
        "if global_step not in capture_steps:"
    )


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
            physics_state=lambda: {
                "num_physics_steps": 1,
                "simulation_time_s": 1.0 / 120.0,
            },
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
            physics_state=lambda: {
                "num_physics_steps": 0,
                "simulation_time_s": 0.0,
            },
        )


def test_same_state_capture_runs_orchestrator_inside_session_and_restores_root():
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

    class Orchestrator:
        def step(self, **kwargs):
            events.append(("step", stage.edit_target, kwargs["delta_time"]))

    class Timeline:
        def get_current_time(self):
            return 0.0

        def is_playing(self):
            return False

    record = sequence._replicator_same_state_step_checked_in_session(
        stage=stage,
        session_layer="session",
        Usd=Usd,
        rep=types.SimpleNamespace(orchestrator=Orchestrator()),
        rt_subframes=64,
        callback_count=lambda: 12,
        timeline=Timeline(),
        physics_state=None,
    )

    assert record["callbacks_before"] == 12
    assert record["callbacks_after"] == 12
    assert stage.edit_target == "root"
    assert events == [
        ("enter", "session"),
        ("step", "session", 0.0),
        ("exit", "session"),
    ]


def test_stage_isosurface_lifecycle_refresh_is_deferred_and_physics_invariant():
    events = []
    proxy_created = [False]
    enabled = [True]
    state = {
        "particle_count": 370,
        "particle_ids_sha256": "1" * 64,
        "positions_sha256": "2" * 64,
    }

    class Timeline:
        playing = True

        def get_current_time(self):
            return 706.0 / 60.0

        def is_playing(self):
            return self.playing

        def pause(self):
            events.append(("pause",))
            self.playing = False

        def set_looping(self, value):
            events.append(("set_looping", value))

        def play(self, *, looping=True):
            events.append(("play", looping))
            self.playing = True

        def commit(self):
            events.append(("commit", self.playing))

    def set_enabled(value):
        events.append(("set_enabled", value))
        enabled[0] = value

    def app_update():
        events.append(("app_update", enabled[0]))

    def source_evidence():
        events.append(("source_evidence",))
        return {
            "mesh_geometry_valid": True,
            "mesh_point_count": 900,
            "mesh_face_count": 1700,
        }

    def ensure_proxy(step):
        events.append(("ensure_proxy", step))
        proxy_created[0] = True

    refresh = getattr(sequence, "_refresh_stage_isosurface_lifecycle", None)
    assert callable(refresh)
    record = refresh(
        timeline=Timeline(),
        callback_count=lambda: 706,
        state_snapshot=lambda: dict(state),
        set_isosurface_enabled=set_enabled,
        read_isosurface_enabled=lambda: enabled[0],
        app_update=app_update,
        source_evidence=source_evidence,
        ensure_proxy=ensure_proxy,
        proxy_created=lambda: proxy_created[0],
        physics_step=706,
        phase="t200-stage",
    )

    assert events == [
        ("pause",),
        ("commit", False),
        ("set_enabled", False),
        ("app_update", False),
        ("set_enabled", True),
        ("app_update", True),
        ("app_update", True),
        ("source_evidence",),
        ("set_looping", False),
        ("play", False),
        ("commit", True),
        ("ensure_proxy", 706),
    ]
    assert record["purpose"] == "stage_isosurface_lifecycle_refresh"
    assert (
        record["documentation_classification"]
        == "installed_nvidia_test_source_plus_bounded_adaptation"
    )
    assert record["documented_initial_creation_update_count"] == 2
    assert record["documented_disable_toggle_update_count"] == 1
    assert record["documented_reenable_toggle_update_count"] == 1
    assert record["experiment_disable_update_count"] == 1
    assert record["experiment_reenable_update_count"] == 2
    assert record["disable_app_update_calls"] == 1
    assert record["disable_app_updates_completed"] == 1
    assert record["reenable_app_update_calls"] == 2
    assert record["reenable_app_updates_completed"] == 2
    assert record["callbacks_before"] == record["callbacks_after"] == 706
    assert record["timeline_time_before_s"] == record["timeline_time_after_s"]
    assert record["timeline_playing_before"] is True
    assert record["timeline_playing_after"] is True
    assert record["isosurface_enabled_readback_before"] is True
    assert record["isosurface_enabled_readback_after_disable"] is False
    assert record["isosurface_enabled_readback_after_enable"] is True
    assert record["particle_state_before"] == state
    assert record["particle_state_after_disable"] == state
    assert record["particle_state_after_enable"] == state
    assert record["particle_state_after_restore"] == state
    assert record["source_isosurface_evidence_after_refresh"]["mesh_geometry_valid"] is True
    assert record["failed_predicates"] == []
    assert record["proxy_created_after_refresh"] is True
    assert record["passed"] is True


def test_stage_isosurface_lifecycle_refresh_fails_closed_on_particle_drift():
    state = {
        "particle_count": 370,
        "particle_ids_sha256": "1" * 64,
        "positions_sha256": "2" * 64,
    }
    enabled = [True]
    proxy_calls = []

    class Timeline:
        playing = True

        def get_current_time(self):
            return 706.0 / 60.0

        def is_playing(self):
            return self.playing

        def pause(self):
            self.playing = False

        def set_looping(self, _value):
            pass

        def play(self, *, looping=True):
            self.playing = True

        def commit(self):
            pass

    def app_update():
        state["positions_sha256"] = "3" * 64

    record = sequence._refresh_stage_isosurface_lifecycle(
        timeline=Timeline(),
        callback_count=lambda: 706,
        state_snapshot=lambda: dict(state),
        set_isosurface_enabled=lambda value: enabled.__setitem__(0, value),
        read_isosurface_enabled=lambda: enabled[0],
        app_update=app_update,
        source_evidence=lambda: {
            "mesh_geometry_valid": True,
            "mesh_point_count": 900,
            "mesh_face_count": 1700,
        },
        ensure_proxy=lambda step: proxy_calls.append(step),
        proxy_created=lambda: False,
        physics_step=706,
        phase="t200-stage",
    )

    assert "particle_state_changed_after_disable" in record["failed_predicates"]
    assert "particle_state_changed_after_enable" in record["failed_predicates"]
    assert "particle_state_changed_after_restore" in record["failed_predicates"]
    assert record["timeline_playing_after"] is True
    assert record["proxy_created_after_refresh"] is False
    assert record["passed"] is False
    assert proxy_calls == []


def test_stage_isosurface_lifecycle_refresh_rejects_degenerate_mesh():
    enabled = [True]
    updates = []
    proxy_calls = []
    state = {
        "particle_count": 370,
        "particle_ids_sha256": "1" * 64,
        "positions_sha256": "2" * 64,
    }

    class Timeline:
        playing = True

        def get_current_time(self):
            return 706.0 / 60.0

        def is_playing(self):
            return self.playing

        def pause(self):
            self.playing = False

        def set_looping(self, _value):
            pass

        def play(self, *, looping=True):
            self.playing = True

        def commit(self):
            pass

    record = sequence._refresh_stage_isosurface_lifecycle(
        timeline=Timeline(),
        callback_count=lambda: 706,
        state_snapshot=lambda: dict(state),
        set_isosurface_enabled=lambda value: enabled.__setitem__(0, value),
        read_isosurface_enabled=lambda: enabled[0],
        app_update=lambda: updates.append(enabled[0]),
        source_evidence=lambda: {
            "mesh_geometry_valid": False,
            "mesh_point_count": 1,
            "mesh_face_count": 1,
            "mesh_nonzero_area_face_count": 0,
        },
        ensure_proxy=lambda step: proxy_calls.append(step),
        proxy_created=lambda: False,
        physics_step=706,
        phase="t200-stage",
    )

    assert updates == [False, True, True]
    assert record["failed_predicates"] == [
        "substantive_isosurface_not_observed",
        "exact_proxy_not_created",
    ]
    assert record["proxy_created_after_refresh"] is False
    assert record["passed"] is False
    assert proxy_calls == []


def test_stage_isosurface_lifecycle_refresh_records_update_exception_honestly():
    enabled = [True]
    state = {
        "particle_count": 370,
        "particle_ids_sha256": "1" * 64,
        "positions_sha256": "2" * 64,
    }

    class Timeline:
        playing = True

        def get_current_time(self):
            return 706.0 / 60.0

        def is_playing(self):
            return self.playing

        def pause(self):
            self.playing = False

        def set_looping(self, _value):
            pass

        def play(self, *, looping=True):
            self.playing = True

        def commit(self):
            pass

    def failed_update():
        raise RuntimeError("bounded update failed")

    record = sequence._refresh_stage_isosurface_lifecycle(
        timeline=Timeline(),
        callback_count=lambda: 706,
        state_snapshot=lambda: dict(state),
        set_isosurface_enabled=lambda value: enabled.__setitem__(0, value),
        read_isosurface_enabled=lambda: enabled[0],
        app_update=failed_update,
        source_evidence=lambda: pytest.fail("mesh evidence must not be inspected"),
        ensure_proxy=lambda step: pytest.fail("proxy must not be created"),
        proxy_created=lambda: False,
        physics_step=706,
        phase="t200-stage",
    )

    assert record["disable_app_update_calls"] == 1
    assert record["disable_app_updates_completed"] == 0
    assert record["reenable_app_update_calls"] == 0
    assert record["reenable_app_updates_completed"] == 0
    assert record["callbacks_after_disable"] is None
    assert record["timeline_time_after_disable_s"] is None
    assert record["particle_state_after_disable"] is None
    assert record["isosurface_enabled_readback_after_disable"] is False
    assert record["isosurface_enabled_readback_after_enable"] is None
    assert "lifecycle_refresh_exception" in record["failed_predicates"]
    assert "disable_observation_missing" in record["failed_predicates"]
    assert record["error"] == "RuntimeError: bounded update failed"
    assert record["passed"] is False


def test_back_to_back_pour_boundary_ownership_is_inclusive_without_overlap():
    plan = sequence.build_three_pour_plan()

    assert sequence._active_tank_for_step(plan, 706) == "t200"
    assert sequence._active_tank_for_step(plan, 707) == "t400"
    assert sequence._active_tank_for_step(plan, 1311) == "t400"
    assert sequence._active_tank_for_step(plan, 1312) == "t100"
    assert sequence._active_tank_for_step(plan, 1984) == "t100"
    assert sequence._active_tank_for_step(plan, 1985) is None


def test_semantic_annotator_precedes_point_instancer_capture_but_follows_stream_labels():
    source = inspect.getsource(sequence.run_sequence)

    label_offset = source.index("system_semantics = UsdSemantics.LabelsAPI.Apply")
    point_label_offset = source.index(
        "_author_dual_liquid_semantics(instancer_prim"
    )
    semantic_attach_offset = source.index(
        "semantic_annotator = rep.AnnotatorRegistry.get_annotator("
    )

    assert "attach_stage" not in source
    stream_label_offset = source.index(
        "stream_semantics = UsdSemantics.LabelsAPI.Apply"
    )

    assert label_offset < stream_label_offset < semantic_attach_offset
    assert point_label_offset < label_offset < stream_label_offset < semantic_attach_offset
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
    assert sum(event["particle_count"] for event in flattened) == plan["totals"]["particle_count"]
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
        sequence.mix_extinction_transmission([[1 / 6, 1 / 6, 4 / 6]])[0],
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


def test_count_driven_mixed_body_uses_frozen_beer_law_transmission(monkeypatch):
    """The live body must use the same nonlinear rule as the frozen stage state."""

    class Attr:
        def __init__(self, value=None):
            self.value = value

        def Set(self, value):
            self.value = value
            return True

        def Get(self):
            return self.value

    class Mesh:
        def __init__(self):
            self.points = Attr()
            self.extent = Attr()
            self.visibility = Attr()

        def GetPointsAttr(self):
            return self.points

        def GetExtentAttr(self):
            return self.extent

        def GetVisibilityAttr(self):
            return self.visibility

    class Shader:
        def __init__(self):
            self.input = Attr()

        def GetInput(self, name):
            assert name == "specular_transmission_color"
            return self.input

    class Gf:
        @staticmethod
        def Vec3f(*values):
            return tuple(float(value) for value in values)

    class Vt:
        @staticmethod
        def Vec3fArray(values):
            return list(values)

    class UsdGeom:
        class Tokens:
            inherited = "inherited"
            invisible = "invisible"

    bottom = sequence.COUNT_DRIVEN_SOURCE_WORLD_BOTTOM_Z_M
    local_points = np.asarray(
        [[0.0, 0.0, bottom], [0.0, 0.0, bottom + 0.01]],
        dtype=np.float64,
    )
    runtime = {
        "source_points": local_points,
        "particle_volume_m3": 1.0e-9,
        "source_world_height_m": 0.01,
        "source_effective_area_m2": 1.0e-6,
        "mesh": Mesh(),
        "face_counts": [],
        "face_indices": [],
    }
    monkeypatch.setattr(
        sequence,
        "build_count_driven_liquid_body_geometry",
        lambda *args, **kwargs: {
            "points": local_points.tolist(),
            "target_height_m": 0.01,
        },
    )
    monkeypatch.setattr(
        sequence,
        "_mesh_world_points",
        lambda *args, **kwargs: local_points.copy(),
    )
    monkeypatch.setattr(
        sequence,
        "signed_mesh_volume_m3",
        lambda *args, **kwargs: 2.0e-9,
    )

    observed = sequence._update_count_driven_twin_liquid_body(
        runtime,
        contained_compositions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        global_shader=Shader(),
        Usd=object(),
        UsdGeom=UsdGeom,
        Gf=Gf,
        Vt=Vt,
    )["body_material_color"]

    # Equal Beer-law extinctions produce the geometric mean of transmissions.
    expected = np.sqrt(
        np.asarray(sequence.NOMINAL_TANK_TRANSMISSION_COLORS["t200"])
        * np.asarray(sequence.NOMINAL_TANK_TRANSMISSION_COLORS["t400"])
    )
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-7)


def test_count_driven_runner_authors_one_persistent_labeled_stream_and_rebinds_material(
    tmp_path,
):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        [
            "--out",
            "unused",
            "--stage",
            str(stage),
            "--count-driven-twin-liquid-mesh",
        ]
    )

    receipt = sequence._empty_paint_ball_receipt(args)
    run_source = inspect.getsource(sequence.run_sequence)
    author_block = run_source[
        run_source.index('failure_phase = "author_render_streams"') :
        run_source.index('failure_phase = "author_vial_colliders"')
    ]
    state_block = run_source[
        run_source.index("def set_stream_state(") :
        run_source.index("def capture_hidden_particle_control(")
    ]

    assert receipt["semantics"]["allowed_stream_paths"] == [
        "/FillStation/FluidmixF2/RenderStream"
    ]
    assert "for tank_id in COMPOSITION_COMPONENTS" not in author_block
    assert author_block.count("UsdGeom.Cylinder.Define") == 1
    assert "UsdSemantics.LabelsAPI.Apply" in author_block
    assert "ComputeBoundMaterial" in state_block
    assert "requested_material_path" in state_block
    assert "readback_material_path" in state_block


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

    with pytest.raises(ValueError, match="video_fps|capture_stride"):
        sequence._validate_args(args)


def test_nuisance_receipt_requires_an_explicit_case_id(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    nuisance = tmp_path / "receipt.json"
    nuisance.write_text("{}", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        [
            "--out",
            str(tmp_path / "out"),
            "--stage",
            str(stage),
            "--nuisance-receipt",
            str(nuisance),
            "--nuisance-receipt-sha256",
            "0" * 64,
        ]
    )

    with pytest.raises(ValueError, match="nuisance-case-id"):
        sequence._validate_args(args)


def test_runtime_calls_nuisance_authoring_after_source_parity_capture():
    source = inspect.getsource(sequence.run_sequence)
    parity = source.index("receipt[\"pre_pour_parity\"].update")
    author = source.index("apply_nuisance_session_layer(")
    assert parity < author
    assert "nuisance_runtime" in source


def test_receipt_predeclares_honest_fallback_claims(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", str(tmp_path / "out"), "--stage", str(stage)]
    )

    receipt = sequence._empty_receipt(args)

    plan = sequence.build_three_pour_plan()
    assert receipt["schema_version"] == 15
    assert receipt["fallback_evidence"] == plan["fallback_trigger"]
    assert receipt["material"]["vial_representation"] == "exact_standard_mesh_proxy_one_global_material"
    assert receipt["material"]["stream_representation"] == "render_only_mesh_current_tank_material"
    assert (
        receipt["material"]["global_blend_rule"]
        == receipt["global_blend"]["global_blend_rule"]
        == plan["fallback_render_model"]["global_blend_rule"]
    )
    assert receipt["stream_render"]["records"] == []
    assert receipt["global_blend"]["model"] == "timeline_owned_runtime_shared_points_stable_usd_ids_pbd_motion_composition_diffusion_stream_prefix_proxy_global_blend_fallback_v5"
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
    assert receipt["mesh_proxy"]["stage_surface_sync_records"] == []
    assert receipt["mesh_proxy"]["stage_isosurface_lifecycle_refresh_records"] == []


def test_receipt_predeclares_ccd_and_internal_composition_evidence_contract(tmp_path):
    stage = tmp_path / "fill_station.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    args = sequence.build_parser().parse_args(
        ["--out", str(tmp_path / "out"), "--stage", str(stage)]
    )

    receipt = sequence._empty_receipt(args)

    assert receipt["schema_version"] == 15
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
    assert "PointInstancer" in source
    assert "_diffuse_active_particle_rows" in source
    assert "fully-mixed" not in source

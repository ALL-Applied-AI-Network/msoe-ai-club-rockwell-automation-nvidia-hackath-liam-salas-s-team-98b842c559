from types import SimpleNamespace

import numpy as np
import pytest


RUN_CONTEXT = {
    "operator_run_id": "run-17",
    "recipe_id": "5",
    "tray_id": "tray-A",
}


@pytest.mark.parametrize("flag", [None, "true", "01"])
def test_legacy_cohort_ids_are_exact_unless_rolling_is_exactly_enabled(
    monkeypatch, flag,
):
    from fluidmix.pipeline.service.production_factory import build_session_cohort

    if flag is None:
        monkeypatch.delenv("FLUIDMIX_ROLLING_COHORTS", raising=False)
    else:
        monkeypatch.setenv("FLUIDMIX_ROLLING_COHORTS", flag)

    context, vials = build_session_cohort(
        RUN_CONTEXT, frame_rate_hz=25.0, cohort_ordinal=7,
    )

    assert context["cohort_id"] == "run-17:tray-A"
    assert [vial["vial_id"] for vial in vials] == [
        "run-17:tray-A:vial-1",
        "run-17:tray-A:vial-2",
        "run-17:tray-A:vial-3",
    ]


def test_rolling_cohort_ids_include_unique_one_based_ordinal(monkeypatch):
    from fluidmix.pipeline.service.production_factory import build_session_cohort

    monkeypatch.setenv("FLUIDMIX_ROLLING_COHORTS", "1")

    first_context, first_vials = build_session_cohort(
        RUN_CONTEXT, frame_rate_hz=10.0,
    )
    second_context, second_vials = build_session_cohort(
        RUN_CONTEXT, frame_rate_hz=10.0, cohort_ordinal=2,
    )

    assert first_context["cohort_id"] == "run-17:tray-A:cohort-1"
    assert second_context["cohort_id"] == "run-17:tray-A:cohort-2"
    assert {vial["vial_id"] for vial in first_vials}.isdisjoint(
        vial["vial_id"] for vial in second_vials
    )


@pytest.mark.parametrize("ordinal", [None, True, False, 0, -1, 1.0, "2"])
def test_rolling_factory_rejects_invalid_private_cohort_ordinal_early(
    monkeypatch, ordinal,
):
    from fluidmix.pipeline.service import production_factory as module

    monkeypatch.setenv("FLUIDMIX_ROLLING_COHORTS", "1")
    factory = object.__new__(module.ProductionAdapterFactory)

    with pytest.raises(module.ProductionFactoryError, match="cohort ordinal"):
        factory.for_session({
            "source": {"kind": "file"},
            "run_context": RUN_CONTEXT,
            "_rolling_cohort_ordinal": ordinal,
        }, "cpu")


def test_factory_passes_rolling_identity_without_changing_recipe_or_tank_settings(
    monkeypatch,
):
    from fluidmix.pipeline.service import production_factory as module
    from fluidmix.perception.tracker import fill_step_production as tracker_module
    from fluidmix.pipeline import rgb_runtime
    from fluidmix.perception.cnn import cnn_predictor, fill_level_adapter

    captured = {}

    class Model:
        def to(self, _device):
            return self

    class IncrementalTracker:
        def __init__(self, **kwargs):
            captured["incremental"] = kwargs

    class Adapter:
        def __init__(self, **kwargs):
            captured["adapter"] = kwargs

    freeze = SimpleNamespace(tank_lab_centres={
        "t200": (0.0, 0.0, 0.0),
        "t400": (1.0, 1.0, 1.0),
        "t100": (2.0, 2.0, 2.0),
    })
    monkeypatch.setenv("FLUIDMIX_ROLLING_COHORTS", "1")
    monkeypatch.setattr(module, "AcceptedPipelineAdapter", Adapter)
    monkeypatch.setattr(tracker_module, "IncrementalFillStepTracker", IncrementalTracker)
    monkeypatch.setattr(tracker_module, "load_pixel_to_metre_calibration", lambda *_a, **_k: object())
    monkeypatch.setattr(tracker_module, "load_topology_freeze", lambda *_a, **_k: object())
    monkeypatch.setattr(tracker_module, "load_tracker_freeze", lambda *_a, **_k: freeze)
    monkeypatch.setattr(rgb_runtime, "load_frozen_cnn", lambda **_k: SimpleNamespace(model=Model()))
    monkeypatch.setattr(cnn_predictor, "predict_pair", lambda *_a, **_k: object())
    monkeypatch.setattr(fill_level_adapter, "configured_weights", lambda: None)

    factory = object.__new__(module.ProductionAdapterFactory)
    factory._snapshots = {}
    factory._vlm_runner = lambda _built: ()
    factory.for_session({
        "source": {"kind": "file"},
        "run_context": RUN_CONTEXT,
        "_rolling_cohort_ordinal": 2,
    }, "cpu")

    tracker = captured["adapter"]["tracker"]
    reference = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker._builder(0, reference)
    settings = captured["incremental"]

    assert settings["cohort_id"] == "run-17:tray-A:cohort-2"
    assert settings["case_id"] == "run-17:tray-A:cohort-2:vial-1"
    assert [(pour.tank_id, pour.dispense_ml) for pour in settings["recipe_pours"]] == [
        ("t200", 10.0), ("t400", 10.0), ("t100", 10.0),
    ]
    assert settings["previous_final_tank_id"] == "t100"
    assert settings["same_run_empty_reference_rgb"] is reference


def test_fresh_session_tracker_uses_manager_current_frame_as_empty_reference():
    from fluidmix.pipeline.service.production_factory import SessionVialTracker

    references = []
    tracker = SessionVialTracker(
        lambda _index, reference: references.append(reference.copy()) or SimpleNamespace(
            observe=lambda *_a, **_k: (),
        ),
        stream_state=lambda *_args: "t200",
    )
    completing_frame = np.full((4, 4, 3), 73, dtype=np.uint8)
    tracker.observe(completing_frame, frame_index=100, elapsed_seconds=10.0)
    tracker.observe(
        np.full((4, 4, 3), 90, dtype=np.uint8),
        frame_index=101,
        elapsed_seconds=10.1,
    )

    assert len(references) == 1
    assert np.array_equal(references[0], completing_frame)

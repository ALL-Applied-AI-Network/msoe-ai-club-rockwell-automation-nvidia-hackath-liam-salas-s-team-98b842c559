from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.cohorts.one_run_manifest_v1 import (
    build_one_run_source_artifacts,
    rebuild_bound_one_run_artifacts,
    validate_one_run_source_spec,
)
from twin.physx.run_variation import (
    bind_tank_colour_resolution_to_artifacts,
    build_tank_colour_resolution,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def source_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_type": "fluidmix-one-run-source-spec-v1",
        "run_id": "RUN-DYNAMIC-AAB-001",
        "split_role": "train",
        "split_version": "dynamic-one-run-v1",
        "source_sha256": {
            "recipe_manifest": "1" * 64,
            "split_spec": "2" * 64,
        },
        "recipes": [
            {
                "recipe_id": "A",
                "programmed_ml": {"t100": 20.0, "t200": 20.0, "t400": 20.0},
            },
            {
                "recipe_id": "B",
                "programmed_ml": {"t100": 10.0, "t200": 30.0, "t400": 20.0},
            },
        ],
        "initial_condition": "clean_start",
        "initial_predecessor_recipe_id": None,
        "vial_recipe_ids": ["A", "A", "B"],
        "residual_volume_ml_by_vial": [0.0, 0.0, 9.0],
        "render_seed_by_vial": [720001, 720002, 720003],
        "nuisance_group_id_by_vial": [
            "DYNAMIC-AAB-V1",
            "DYNAMIC-AAB-V2",
            "DYNAMIC-AAB-V3",
        ],
    }


def test_builder_normalizes_and_embeds_one_run_source_spec():
    spec = source_spec()

    assert validate_one_run_source_spec(spec) == spec
    artifacts = build_one_run_source_artifacts(spec)
    plan = artifacts["render-schedule-plan.json"]
    model = artifacts["model-input-plan.json"]
    labels = artifacts["label-provenance.json"]

    assert plan["artifact_type"] == "physx_f2_source_bound_one_run_render_plan_v1"
    assert plan["source_run_spec"] == spec
    assert plan["source_run_spec_sha256"] == _canonical_sha256(spec)
    assert len(plan["runs"]) == len(plan["batches"]) == 1
    assert len(plan["execution_schedules"]) == 3
    assert model["artifact_type"] == "synthetic_f2_source_bound_one_run_input_plan_v1"
    assert labels["artifact_type"] == (
        "synthetic_f2_source_bound_one_run_label_provenance_v1"
    )
    assert [
        row["case_binding"]["current_recipe_id"]
        for row in plan["execution_schedules"]
    ] == ["A", "A", "B"]
    assert [
        row["case_binding"]["previous_recipe_id"]
        for row in plan["execution_schedules"]
    ] == [None, "A", "A"]


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda value: value.__setitem__("extra", True), "unexpected fields"),
        (
            lambda value: value["source_sha256"].__setitem__("extra", "3" * 64),
            "source_sha256",
        ),
        (
            lambda value: value["recipes"].append(deepcopy(value["recipes"][0])),
            "recipe definitions",
        ),
        (
            lambda value: value["vial_recipe_ids"].__setitem__(2, "MISSING"),
            "vial recipe",
        ),
        (
            lambda value: value["render_seed_by_vial"].__setitem__(0, True),
            "render seed",
        ),
        (
            lambda value: value["residual_volume_ml_by_vial"].__setitem__(0, 1.0),
            "clean start",
        ),
        (
            lambda value: value["nuisance_group_id_by_vial"].pop(),
            "parallel vial arrays",
        ),
    ],
)
def test_source_spec_rejects_invalid_contract(mutate, error):
    spec = source_spec()
    mutate(spec)
    with pytest.raises((TypeError, ValueError), match=error):
        validate_one_run_source_spec(spec)


def test_bound_rebuild_recreates_all_three_artifacts_without_catalog(monkeypatch):
    base = build_one_run_source_artifacts(source_spec())
    resolution = build_tank_colour_resolution("amber_teal")
    bound = bind_tank_colour_resolution_to_artifacts(base, resolution)

    def forbid_catalog(*_args, **_kwargs):
        raise AssertionError("catalog must not be reopened")

    monkeypatch.setattr(
        "twin.physx.run_variation.load_tank_colour_preset_catalog",
        forbid_catalog,
    )
    assert rebuild_bound_one_run_artifacts(
        bound["render-schedule-plan.json"]
    ) == bound


@pytest.mark.parametrize(
    "mutate",
    [
        lambda artifacts: artifacts["render-schedule-plan.json"][
            "source_run_spec"
        ]["vial_recipe_ids"].__setitem__(2, "A"),
        lambda artifacts: artifacts["render-schedule-plan.json"].__setitem__(
            "source_run_spec_sha256", "0" * 64
        ),
        lambda artifacts: artifacts["render-schedule-plan.json"]["runs"][0][
            "palette"
        ].__setitem__("palette_sha256", "0" * 64),
        lambda artifacts: artifacts["render-schedule-plan.json"]["batches"][0].__setitem__(
            "palette_sha256", "0" * 64
        ),
        lambda artifacts: artifacts["render-schedule-plan.json"][
            "execution_schedules"
        ][0]["case_binding"].__setitem__("palette_sha256", "0" * 64),
        lambda artifacts: artifacts["render-schedule-plan.json"][
            "execution_schedule_sha256_by_case"
        ].update({next(iter(artifacts["render-schedule-plan.json"]["execution_schedule_sha256_by_case"])): "0" * 64}),
    ],
)
def test_bound_rebuild_rejects_graph_mutation(mutate):
    base = build_one_run_source_artifacts(source_spec())
    bound = bind_tank_colour_resolution_to_artifacts(
        base, build_tank_colour_resolution("amber_teal")
    )
    mutate(bound)
    plan = bound["render-schedule-plan.json"]
    plan_hash = _canonical_sha256(plan)
    assert len(plan_hash) == 64

    with pytest.raises((TypeError, ValueError)):
        rebuild_bound_one_run_artifacts(plan)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda artifacts: artifacts["label-provenance.json"]["vial_labels"][0].update(
            render_schedule_sha256="0" * 64
        ),
        lambda artifacts: artifacts["label-provenance.json"].update(
            tank_colour_resolution=build_tank_colour_resolution("base")
        ),
        lambda artifacts: artifacts["model-input-plan.json"]["vial_records"][0].update(
            renderer_palette_sha256="0" * 64
        ),
    ],
)
def test_public_binder_rejects_bound_label_or_model_mutation(mutate):
    resolution = build_tank_colour_resolution("amber_teal")
    bound = bind_tank_colour_resolution_to_artifacts(
        build_one_run_source_artifacts(source_spec()), resolution
    )
    mutate(bound)

    with pytest.raises((TypeError, ValueError)):
        bind_tank_colour_resolution_to_artifacts(bound, resolution)

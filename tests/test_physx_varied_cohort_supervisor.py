from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "varied_cohort_supervisor",
    ROOT / "scripts" / "run_physx_train_residual_cohort.py",
)
assert SPEC is not None and SPEC.loader is not None
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)

from fluidmix.physx_varied_cohort_manifest_v3 import (  # noqa: E402
    build_varied_cohort_artifacts,
)
from fluidmix.physx_varied_cohort_manifest_v4 import (  # noqa: E402
    build_varied_cohort_artifacts as build_varied_cohort_artifacts_v4,
)
from fluidmix.physx_train_residual_manifest_v2 import (  # noqa: E402
    build_train_residual_artifacts,
)
from fluidmix.physx_run_variation import (  # noqa: E402
    build_tank_colour_plan_artifacts,
    build_tank_colour_resolution,
    bind_tank_colour_resolution_to_artifacts,
)
from fluidmix.physx_one_run_manifest_v1 import (  # noqa: E402
    build_one_run_source_artifacts,
)


def _new_one_run_artifacts() -> dict[str, dict[str, object]]:
    case_id = "CASE_SOURCE_BOUND"
    schedule = {
        "case_binding": {
            "case_id": case_id,
            "batch_id": "BATCH_SOURCE_BOUND",
            "run_id": "RUN_SOURCE_BOUND",
            "palette_sha256": "0" * 64,
        },
        "model_input_record": {
            "recipe_ml": {"t100": 20, "t200": 10, "t400": 30}
        },
    }
    return {
        "render-schedule-plan.json": {
            "artifact_type": "new-source-bound-one-run-plan-v1",
            "runs": [
                {
                    "run_id": "RUN_SOURCE_BOUND",
                    "case_ids": [case_id],
                    "palette": {"palette_sha256": "0" * 64},
                }
            ],
            "batches": [
                {
                    "batch_id": "BATCH_SOURCE_BOUND",
                    "run_id": "RUN_SOURCE_BOUND",
                    "case_ids": [case_id],
                    "palette_sha256": "0" * 64,
                }
            ],
            "execution_schedules": [schedule],
            "execution_schedule_sha256_by_case": {
                case_id: _canonical_sha256(schedule)
            },
        },
        "model-input-plan.json": {
            "vial_records": [
                {
                    "case_id": case_id,
                    "run_id": "RUN_SOURCE_BOUND",
                    "recipe_ml": {"t100": 20, "t200": 10, "t400": 30},
                }
            ]
        },
        "label-provenance.json": {
            "vial_labels": [
                {
                    "case_id": case_id,
                    "render_schedule_sha256": _canonical_sha256(schedule),
                }
            ]
        },
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _dynamic_aab_artifacts() -> dict[str, dict[str, object]]:
    base = build_one_run_source_artifacts(
        {
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
            "nuisance_group_id_by_vial": ["AAB-V1", "AAB-V2", "AAB-V3"],
        }
    )
    return bind_tank_colour_resolution_to_artifacts(
        base, build_tank_colour_resolution("amber_teal")
    )


def test_dynamic_one_run_plan_rebuilds_through_supervisor_loader(tmp_path, monkeypatch):
    artifacts = _dynamic_aab_artifacts()
    plan = artifacts["render-schedule-plan.json"]
    path = (tmp_path / "dynamic-plan.json").resolve()
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "fluidmix.physx_run_variation.load_tank_colour_preset_catalog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("catalog reopened")
        ),
    )

    loaded = supervisor._load_bound_plan(path, _canonical_sha256(plan))

    assert loaded == plan
    assert supervisor._source_plan_denominators(loaded)["case_count"] == 3
    assert supervisor._case_output_contract(loaded)["case_count"] == 3
    assert set(supervisor._case_run_palettes(loaded)) == {
        row["case_binding"]["case_id"] for row in plan["execution_schedules"]
    }


def test_dynamic_one_run_plan_rebuilds_through_renderer_loader(tmp_path, monkeypatch):
    artifacts = _dynamic_aab_artifacts()
    plan = artifacts["render-schedule-plan.json"]
    schedule = plan["execution_schedules"][2]
    case_id = schedule["case_binding"]["case_id"]
    path = (tmp_path / "dynamic-plan.json").resolve()
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    stage = (tmp_path / "stage.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    qualification = (tmp_path / "qualification.json").resolve()
    qualification.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        supervisor.sequence,
        "_bind_accepted_qualification_receipt",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "b" * 64,
            "physics_substeps": 1,
            "particle_resolution_profile_id": "c20_2p5mm",
        },
    )
    monkeypatch.setattr(
        "fluidmix.physx_run_variation.load_tank_colour_preset_catalog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("catalog reopened")
        ),
    )
    args = supervisor.sequence.build_parser().parse_args(
        [
            "--out", str((tmp_path / "out").resolve()),
            "--stage", str(stage),
            "--seed", "720003",
            "--count-driven-twin-liquid-mesh",
            "--height-banded-body",
            "--realtime-video",
            "--train-render-plan", str(path),
            "--train-render-plan-sha256", _canonical_sha256(plan),
            "--train-case-id", case_id,
            "--accepted-qualification-receipt", str(qualification),
            "--accepted-qualification-receipt-sha256", "b" * 64,
        ]
    )

    contract = supervisor.sequence._execution_schedule_contract(args)
    runtime = supervisor.sequence._runtime_paint_ball_plan(args)

    assert contract["contract_kind"] == "source_bound_one_run_v1"
    assert contract["source_run_spec_sha256"] == plan["source_run_spec_sha256"]
    assert contract["tank_colour_resolution"] == plan["tank_colour_resolution"]
    assert contract["model_input_record"] == artifacts["model-input-plan.json"][
        "vial_records"
    ][2]
    assert runtime["execution_schedule"]["contract_kind"] == (
        "source_bound_one_run_v1"
    )
    assert runtime["tank_colour_resolution"] == plan["tank_colour_resolution"]


def test_dynamic_one_run_renderer_main_plan_only_uses_loaded_resolution(
    tmp_path, monkeypatch
):
    artifacts = _dynamic_aab_artifacts()
    plan = artifacts["render-schedule-plan.json"]
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    plan_path = (tmp_path / "dynamic-plan.json").resolve()
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    stage = (tmp_path / "stage.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    qualification = (tmp_path / "qualification.json").resolve()
    qualification.write_text("{}\n", encoding="utf-8")
    output = (tmp_path / "plan-only").resolve()
    monkeypatch.setattr(
        supervisor.sequence,
        "_bind_accepted_qualification_receipt",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "b" * 64,
            "physics_substeps": 1,
            "particle_resolution_profile_id": "c20_2p5mm",
        },
    )

    exit_code = supervisor.sequence.main(
        [
            "--out", str(output),
            "--stage", str(stage),
            "--seed", "720001",
            "--water-pbd-preset",
            "--count-driven-twin-liquid-mesh",
            "--solver-position-iterations", "16",
            "--height-banded-body",
            "--realtime-video",
            "--train-render-plan", str(plan_path),
            "--train-render-plan-sha256", _canonical_sha256(plan),
            "--train-case-id", case_id,
            "--accepted-qualification-receipt", str(qualification),
            "--accepted-qualification-receipt-sha256", "b" * 64,
            "--plan-only",
        ]
    )

    runtime = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    labels = json.loads(
        (output / "label-provenance.json").read_text(encoding="utf-8")
    )
    assert exit_code == 0
    assert runtime["tank_colour_resolution"] == plan["tank_colour_resolution"]
    assert runtime["execution_schedule"]["source_run_spec_sha256"] == plan[
        "source_run_spec_sha256"
    ]
    assert labels == {
        "schema_version": 1,
        "artifact_type": "tank_colour_label_provenance",
        "tank_colour_resolution": plan["tank_colour_resolution"],
        "excluded_from_model_inputs": True,
    }


@pytest.mark.parametrize(
    "appearance_args",
    (
        ["--tank-colours", "base"],
        ["--run-palette-id", "LEGACY", "--run-palette-seed", "1"],
    ),
)
def test_dynamic_one_run_renderer_rejects_direct_or_legacy_palette_options(
    tmp_path, monkeypatch, appearance_args
):
    plan = _dynamic_aab_artifacts()["render-schedule-plan.json"]
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    plan_path = (tmp_path / "dynamic-plan.json").resolve()
    plan_path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    stage = (tmp_path / "stage.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    qualification = (tmp_path / "qualification.json").resolve()
    qualification.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        supervisor.sequence,
        "_bind_accepted_qualification_receipt",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "b" * 64,
            "physics_substeps": 1,
            "particle_resolution_profile_id": "c20_2p5mm",
        },
    )

    with pytest.raises(SystemExit):
        supervisor.sequence.main(
            [
                "--out", str((tmp_path / "must-not-exist").resolve()),
                "--stage", str(stage),
                "--seed", "720001",
                "--water-pbd-preset",
                "--count-driven-twin-liquid-mesh",
                "--solver-position-iterations", "16",
                "--height-banded-body",
                "--realtime-video",
                "--train-render-plan", str(plan_path),
                "--train-render-plan-sha256", _canonical_sha256(plan),
                "--train-case-id", case_id,
                "--accepted-qualification-receipt", str(qualification),
                "--accepted-qualification-receipt-sha256", "b" * 64,
                *appearance_args,
                "--plan-only",
            ]
        )


def test_supervisor_accepts_one_case_shards_for_the_v4_schedule_count():
    plan = {
        "execution_schedules": [
            {"case_binding": {"case_id": f"CASE_{index:02d}"}}
            for index in range(30)
        ]
    }
    selected = supervisor._selected_schedule_rows(
        plan, shard_count=30, shard_index=29
    )
    assert [row[1]["case_binding"]["case_id"] for row in selected] == [
        "CASE_29"
    ]


def test_supervisor_rejects_shards_above_selected_source_schedule_count():
    plan = {
        "execution_schedules": [
            {"case_binding": {"case_id": f"CASE_{index:02d}"}}
            for index in range(30)
        ]
    }
    with pytest.raises(ValueError, match="between 1 and 30"):
        supervisor._selected_schedule_rows(plan, shard_count=31, shard_index=0)


def test_supervisor_source_rebuilds_v4_and_derives_30_case_denominators(tmp_path):
    plan = build_varied_cohort_artifacts_v4(repo_root=ROOT)["render-schedule-plan.json"]
    path = (tmp_path / "plan-v4.json").resolve()
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    digest = _canonical_sha256(plan)

    loaded = supervisor._load_bound_plan(path, digest)
    output = supervisor._case_output_contract(loaded)
    palettes = supervisor._case_run_palettes(loaded)

    assert loaded == plan
    assert output["case_count"] == 30
    assert output["keep_count"] == 15
    assert output["discard_count"] == 15
    assert output["run_count"] == 10
    assert output["batch_count"] == 10
    assert output["split_counts"] == {
        "train": 12,
        "validation": 9,
        "evaluation": 9,
    }
    assert len(palettes) == 30


def test_supervisor_rejects_unknown_source_artifact_type(tmp_path):
    plan = build_varied_cohort_artifacts(repo_root=ROOT)["render-schedule-plan.json"]
    plan["artifact_type"] = "physx_f2_varied_run_residual_render_plan_unknown"
    path = (tmp_path / "unknown-plan.json").resolve()
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not source-bound"):
        supervisor._load_bound_plan(path, _canonical_sha256(plan))


def test_renderer_rejects_unknown_source_artifact_type_without_v2_fallback(tmp_path):
    plan = build_varied_cohort_artifacts(repo_root=ROOT)["render-schedule-plan.json"]
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    plan["artifact_type"] = "physx_f2_unknown_source_plan"
    path = (tmp_path / "unknown-renderer-plan.json").resolve()
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact type is not source-bound"):
        supervisor.sequence._load_train_residual_execution_contract(
            path,
            _canonical_sha256(plan),
            case_id,
            (tmp_path / "qualification.json").resolve(),
            "b" * 64,
            stage_path=(tmp_path / "stage.usda").resolve(),
            rt_subframes=64,
            pathtracing_total_spp=64,
            count_driven_twin_liquid_mesh=True,
            particle_resolution_profile_id="c20_2p5mm",
        )


def test_supervisor_loads_source_rebuilt_varied_plan_and_computes_realtime_counts(tmp_path):
    plan = build_varied_cohort_artifacts(repo_root=ROOT)["render-schedule-plan.json"]
    path = (tmp_path / "plan.json").resolve()
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    digest = _canonical_sha256(plan)

    loaded = supervisor._load_bound_plan(path, digest)
    output = supervisor._case_output_contract(loaded)
    palettes = supervisor._case_run_palettes(loaded)

    assert loaded == plan
    assert output["case_count"] == 18
    assert output["keep_count"] == 12
    assert output["discard_count"] == 6
    assert output["source_capture_count"] == 42
    assert output["decoded_video_frame_count"] > output["source_capture_count"]
    assert len(palettes) == 18
    assert len({row["run_palette_id"] for row in palettes.values()}) == 6


def test_varied_case_command_enables_realtime_bands_and_bound_run_palette(tmp_path):
    plan = build_varied_cohort_artifacts(repo_root=ROOT)["render-schedule-plan.json"]
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    palettes = supervisor._case_run_palettes(plan)
    args = argparse.Namespace(
        python_executable=(tmp_path / "python").resolve(),
        renderer_script=(tmp_path / "renderer.py").resolve(),
        stage=(tmp_path / "stage.usda").resolve(),
        plan=(tmp_path / "plan.json").resolve(),
        plan_sha256="a" * 64,
        accepted_qualification_receipt=(tmp_path / "qualification.json").resolve(),
        accepted_qualification_receipt_sha256="b" * 64,
        nuisance_receipt=(tmp_path / "nuisance.json").resolve(),
        nuisance_receipt_sha256="c" * 64,
        _run_palette_by_case=palettes,
    )
    command = supervisor._case_command(
        args,
        case_id=case_id,
        seed=510101,
        nuisance_case_id="nuisance",
        app_output=(tmp_path / "output").resolve(),
    )

    assert "--realtime-video" in command
    assert "--height-banded-body" in command
    palette = palettes[case_id]
    assert command[command.index("--run-palette-id") + 1] == palette["run_palette_id"]
    assert command[command.index("--run-palette-seed") + 1] == str(
        palette["run_palette_seed"]
    )


@pytest.mark.parametrize(
    ("builder", "expected_command_sha256"),
    [
        (
            build_train_residual_artifacts,
            "5bf92fea5a7e941081c9b56a4e937fd2cdeaa9ecc6ba77d125dff1b257076de0",
        ),
        (build_varied_cohort_artifacts, "0608ab047c2e0cf733649a9b9c0224b6412f7654613dea84af014e3a347d637b"),
        (build_varied_cohort_artifacts_v4, "a86bce8f4e80adc985f6d5501cf3f941c84a962d114d04012949b905ff9ed572"),
    ],
)
def test_legacy_supervisor_command_is_exactly_frozen(builder, expected_command_sha256):
    plan = builder(repo_root=ROOT)["render-schedule-plan.json"]
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    batch = next(row for row in plan["batches"] if case_id in row["case_ids"])
    seed = batch["render_seed_by_vial"][batch["case_ids"].index(case_id)]
    args = argparse.Namespace(
        python_executable=Path("/opt/isaac/python.sh"),
        renderer_script=Path("/repo/scripts/physx_three_pour_sequence.py"),
        stage=Path("/data/stage.usd"),
        plan=Path("/data/plan.json"),
        plan_sha256="a" * 64,
        accepted_qualification_receipt=Path("/data/qualification.json"),
        accepted_qualification_receipt_sha256="b" * 64,
        nuisance_receipt=Path("/data/nuisance.json"),
        nuisance_receipt_sha256="c" * 64,
        _run_palette_by_case=supervisor._case_run_palettes(plan),
    )
    command = supervisor._case_command(
        args,
        case_id=case_id,
        seed=seed,
        nuisance_case_id="nuisance-0",
        app_output=Path("/out/app-output"),
    )
    assert hashlib.sha256(
        json.dumps(command, separators=(",", ":")).encode()
    ).hexdigest() == expected_command_sha256


def test_new_source_plan_uses_stored_resolution_without_catalog_reopen(monkeypatch):
    artifacts = build_tank_colour_plan_artifacts(
        _new_one_run_artifacts(),
        "crimson_indigo",
        randomize_around_preset=True,
        randomization_seed=5801,
        free_text_label="new source plan",
    )
    plan = artifacts["render-schedule-plan.json"]
    monkeypatch.setattr(
        "fluidmix.tank_colour_presets.load_tank_colour_preset_catalog",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("catalog reopened")),
    )

    palettes = supervisor._case_run_palettes(plan)
    case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
    args = argparse.Namespace(
        python_executable=Path("/opt/isaac/python.sh"),
        renderer_script=Path("/repo/scripts/physx_three_pour_sequence.py"),
        stage=Path("/data/stage.usd"),
        plan=Path("/data/plan.json"),
        plan_sha256=_canonical_sha256(plan),
        accepted_qualification_receipt=Path("/data/qualification.json"),
        accepted_qualification_receipt_sha256="b" * 64,
        nuisance_receipt=Path("/data/nuisance.json"),
        nuisance_receipt_sha256="c" * 64,
        _run_palette_by_case=palettes,
    )
    command = supervisor._case_command(
        args,
        case_id=case_id,
        seed=520101,
        nuisance_case_id="nuisance-0",
        app_output=Path("/out/app-output"),
    )

    assert plan["tank_colour_resolution"]["preset_name"] == "crimson_indigo"
    assert palettes[case_id]["palette_sha256"] == plan["run_palette"]["palette_sha256"]
    assert "--height-banded-body" in command
    assert "--realtime-video" in command
    assert "--run-palette-id" not in command
    assert "--tank-colours" not in command


def test_new_source_plan_rejects_renderer_palette_disagreement():
    artifacts = build_tank_colour_plan_artifacts(
        _new_one_run_artifacts(), "base"
    )
    plan = artifacts["render-schedule-plan.json"]
    plan["run_palette"]["palette_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="run_palette"):
        supervisor._case_run_palettes(plan)


def test_renderer_loader_binds_varied_case_to_its_run_palette(tmp_path, monkeypatch):
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    plan = artifacts["render-schedule-plan.json"]
    schedule = plan["execution_schedules"][0]
    case_id = schedule["case_binding"]["case_id"]
    batch = next(row for row in plan["batches"] if case_id in row["case_ids"])
    seed = batch["render_seed_by_vial"][batch["case_ids"].index(case_id)]
    path = (tmp_path / "plan.json").resolve()
    path.write_text(json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8")
    stage = (tmp_path / "stage.usda").resolve()
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    qualification = (tmp_path / "qualification.json").resolve()
    qualification.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        supervisor.sequence,
        "_bind_accepted_qualification_receipt",
        lambda *args, **kwargs: {"receipt_sha256": "b" * 64},
    )
    args = supervisor.sequence.build_parser().parse_args(
        [
            "--out",
            str((tmp_path / "out").resolve()),
            "--stage",
            str(stage),
            "--seed",
            str(seed),
            "--count-driven-twin-liquid-mesh",
            "--height-banded-body",
            "--realtime-video",
            "--run-palette-id",
            batch["run_palette_id"],
            "--run-palette-seed",
            str(batch["run_palette_seed"]),
            "--train-render-plan",
            str(path),
            "--train-render-plan-sha256",
            _canonical_sha256(plan),
            "--train-case-id",
            case_id,
            "--accepted-qualification-receipt",
            str(qualification),
            "--accepted-qualification-receipt-sha256",
            "b" * 64,
        ]
    )

    contract = supervisor.sequence._execution_schedule_contract(args)
    runtime = supervisor.sequence._runtime_paint_ball_plan(args)

    assert contract["contract_kind"] == "varied_run_residual_v3"
    assert contract["run_palette_binding"] == {
        "run_palette_id": batch["run_palette_id"],
        "run_palette_seed": batch["run_palette_seed"],
        "palette_sha256": schedule["case_binding"]["palette_sha256"],
    }
    assert runtime["execution_schedule"]["contract_kind"] == "varied_run_residual_v3"
    assert runtime["execution_schedule"]["adapter_schema_version"] == 3
    assert runtime["run_palette"]["palette_sha256"] == schedule["case_binding"][
        "palette_sha256"
    ]

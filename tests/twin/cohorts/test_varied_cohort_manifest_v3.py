from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.cohorts.varied_cohort_manifest_v3 import (  # noqa: E402
    build_varied_cohort_artifacts,
)


def _canonical_sha256(value: object) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def test_varied_cohort_has_six_whole_run_batches_and_eighteen_vials():
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    render = artifacts["render-schedule-plan.json"]
    model = artifacts["model-input-plan.json"]
    labels = artifacts["label-provenance.json"]

    assert render["artifact_type"] == "physx_f2_varied_run_residual_render_plan_v3"
    assert len(render["runs"]) == 6
    assert len(render["batches"]) == 6
    assert len(render["execution_schedules"]) == 18
    assert render["denominators"] == {
        "runs": 6,
        "batches": 6,
        "vials": 18,
        "train_vials": 12,
        "validation_vials": 3,
        "evaluation_vials": 3,
        "keep_vials": 12,
        "discard_vials": 6,
        "accepted_render_artifacts": 0,
    }
    assert len(model["vial_records"]) == 18
    assert len(labels["vial_labels"]) == 18
    assert {
        name: _canonical_sha256(value) for name, value in artifacts.items()
    } == {
        "render-schedule-plan.json": "108aaa8a3e8922d73f2f9241c474f75e7c0a927049dfc457a7dd64b4acce7b2a",
        "model-input-plan.json": "3e43d991f5c8ca7ecb52c29c5078609e8036e5d4dff43163d41a330999e04ceb",
        "label-provenance.json": "240b11cd2b3ab7ccf0a901029b3c78e797609ebc6eb68d4eb1d00dd26e2bbe29",
    }

    split_by_run: dict[str, set[str]] = {}
    palettes_by_run: dict[str, set[str]] = {}
    for schedule in render["execution_schedules"]:
        binding = schedule["case_binding"]
        split_by_run.setdefault(binding["run_id"], set()).add(binding["split_role"])
        palettes_by_run.setdefault(binding["run_id"], set()).add(
            binding["palette_sha256"]
        )
        assert schedule["physical_sequence"]["premixed_before_vial"] is False
        assert schedule["physical_sequence"]["mixing_location"] == "vial"
        assert schedule["physical_sequence"]["dispense_order"] == [
            tank
            for tank in ("t200", "t400", "t100")
            if schedule["model_input_record"]["programmed_ml"][tank] > 0
        ]
    assert all(len(roles) == 1 for roles in split_by_run.values())
    assert all(len(palettes) == 1 for palettes in palettes_by_run.values())
    assert len({next(iter(values)) for values in palettes_by_run.values()}) == 6


def test_varied_cohort_uses_real_recipe_volumes_and_excludes_oracles_from_model_rows():
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    render = artifacts["render-schedule-plan.json"]
    model = artifacts["model-input-plan.json"]
    labels = artifacts["label-provenance.json"]

    recipe_ids = {
        schedule["case_binding"]["current_recipe_id"]
        for schedule in render["execution_schedules"]
    }
    assert recipe_ids == {"5", "6", "7", "8", "13", "14"}
    assert any(
        0 in schedule["model_input_record"]["programmed_ml"].values()
        for schedule in render["execution_schedules"]
    )
    for run in render["runs"]:
        assert run["palette"]["transmission_1cm"]["t100"] == [1.0, 1.0, 1.0]
        assert set(run["palette"]["variation_by_tank"]) == {"t200", "t400"}

    forbidden = ("residual", "label", "oracle", "disposition", "severity", "palette")
    assert all(
        not any(fragment in key.lower() for fragment in forbidden)
        for row in model["vial_records"]
        for key in row
    )
    dispositions = [row["expected_disposition"] for row in labels["vial_labels"]]
    assert dispositions.count("keep") == 12
    assert dispositions.count("discard") == 6
    assert all(
        row["expected_disposition"]
        == ("keep" if row["prescribed_leading_residual_volume_ml"] == 0 else "discard")
        for row in labels["vial_labels"]
    )

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from tests.fluidmix.storage.artifacts.test_capture_manifest_adapter import _manifest  # noqa: E402


SCRIPT = ROOT / "cli/fluidmix/explanation/vlm" / "build_model_input_plan_from_manifests.py"


def _context(names: tuple[str, ...] = ("TARGET", "REF-A", "REF-B")) -> dict:
    rows = []
    for name in names:
        rows.append(
            {
                "batch_id": name,
                "cohort": "scored_target" if name == "TARGET" else "reference_only",
                "family_id": "VA",
                "current_recipe_id": "S0380",
                "previous_recipe_id": "S0160",
                "run_id": f"run-{name}",
                "split_role": "val",
                "programmed_recipe": {
                    "dispense_order": ["t200", "t400", "t100"],
                    "previous_ml_by_tank": {"t200": 20, "t400": 10, "t100": 30},
                    "current_ml_by_tank": {"t200": 30, "t400": 20, "t100": 10},
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "stage_batch_context",
        "split_version": "inn-7-splits-v2",
        "batches": rows,
    }


def _captures(tmp_path: Path, names: tuple[str, ...] = ("TARGET", "REF-A", "REF-B")) -> list[Path]:
    paths = []
    for name in names:
        root = tmp_path / name
        root.mkdir()
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(_manifest(root, batch_id=name)), encoding="utf-8")
        paths.append(path)
    return paths


def test_programmed_recipe_accepts_zero_tank_with_positive_recipe_totals():
    from fluidmix.explanation.vlm.model_input_plan_builder import _parse_programmed_recipe

    recipe = {
        "dispense_order": ["t200", "t400", "t100"],
        "previous_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
        "current_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
    }

    normalized = _parse_programmed_recipe(recipe)

    assert normalized["previous_ml_by_tank"]["t200"] == 0.0
    assert normalized["current_ml_by_tank"]["t200"] == 0.0


@pytest.mark.parametrize(
    "bad_value",
    [-1, False, float("inf"), float("nan")],
    ids=["negative", "bool", "infinite", "nan"],
)
def test_programmed_recipe_rejects_invalid_tank_volume(bad_value):
    from fluidmix.explanation.vlm.model_input_plan_builder import (
        ModelInputPlanError,
        _parse_programmed_recipe,
    )

    recipe = {
        "dispense_order": ["t200", "t400", "t100"],
        "previous_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
        "current_ml_by_tank": {"t100": 25, "t200": bad_value, "t400": 35},
    }

    with pytest.raises(ModelInputPlanError, match="t200_invalid"):
        _parse_programmed_recipe(recipe)


@pytest.mark.parametrize(
    "recipe_key", ["previous_ml_by_tank", "current_ml_by_tank"]
)
def test_programmed_recipe_rejects_all_zero_recipe(recipe_key):
    from fluidmix.explanation.vlm.model_input_plan_builder import (
        ModelInputPlanError,
        _parse_programmed_recipe,
    )

    recipe = {
        "dispense_order": ["t200", "t400", "t100"],
        "previous_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
        "current_ml_by_tank": {"t100": 25, "t200": 0, "t400": 35},
    }
    recipe[recipe_key] = {"t100": 0, "t200": 0, "t400": 0}

    with pytest.raises(ModelInputPlanError, match="total_invalid"):
        _parse_programmed_recipe(recipe)


def test_builds_three_vial_decisions_with_all_reference_vials(tmp_path):
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    plan = build_model_input_plan(_captures(tmp_path), _context())

    assert plan["artifact_type"] == "model_input_plan"
    assert len(plan["decision_records"]) == 3
    assert all(len(row["stages"]) == 3 for row in plan["decision_records"])
    primary = next(
        case
        for case in plan["decision_records"][0]["stages"][0]["reference_cases"]
        if case["disposition_source"]
    )
    assert primary["mode"] == "all_reference_batches"
    assert {peer["batch_id"] for peer in primary["candidate_peers"]} == {"TARGET", "REF-A", "REF-B"}
    assert len(primary["candidate_peers"]) == 8
    assert "path" not in json.dumps(plan)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda c: c["batches"].__setitem__(2, {**c["batches"][2], "family_id": "TR"}), "context_capture_provenance"),
        (lambda c: c["batches"].pop(), "context_capture_batch_set"),
        (lambda c: c["batches"][0].update({"oracle": "bad"}), "forbidden_oracle"),
    ],
)
def test_rejects_unsafe_context(mutation, expected, tmp_path):
    from fluidmix.explanation.vlm.model_input_plan_builder import ModelInputPlanError, build_model_input_plan

    context = _context()
    mutation(context)
    with pytest.raises(ModelInputPlanError, match=expected):
        build_model_input_plan(_captures(tmp_path), context)


def test_deterministic_for_manifest_argument_order(tmp_path):
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    paths = _captures(tmp_path)
    context = _context()
    left = build_model_input_plan(paths, context)
    right = build_model_input_plan(list(reversed(paths)), context)
    assert left == right


def test_thin_reference_bank_builds_but_deterministic_evidence_abstains(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        bind_stage_capture_manifest,
        build_measured_evidence,
    )
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    names = ("TARGET", "REF-A")
    paths = _captures(tmp_path, names)
    plan = build_model_input_plan(paths, _context(names))
    primary = next(
        case
        for case in plan["decision_records"][0]["stages"][0]["reference_cases"]
        if case["disposition_source"]
    )
    assert {peer["batch_id"] for peer in primary["candidate_peers"]} == {"REF-A"}
    captures = [
        bind_stage_capture_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            base_dir=path.parent / path.stem,
        )
        for path in paths
    ]

    measured = build_measured_evidence(plan, captures)

    assert {row["disposition"] for row in measured["decision_records"]} == {
        "insufficient_evidence"
    }
    assert measured["denominators"]["abstained_vial_decisions"] == 3


def test_same_run_reference_batches_do_not_manufacture_independence(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        bind_stage_capture_manifest,
        build_measured_evidence,
    )
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    paths = _captures(tmp_path)
    context = _context()
    context["batches"][2]["run_id"] = context["batches"][1]["run_id"]
    plan = build_model_input_plan(paths, context)
    primary = next(
        case
        for case in plan["decision_records"][0]["stages"][0]["reference_cases"]
        if case["disposition_source"]
    )
    assert {peer["batch_id"] for peer in primary["candidate_peers"]} == {"REF-A"}
    captures = [
        bind_stage_capture_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            base_dir=path.parent / path.stem,
        )
        for path in paths
    ]

    measured = build_measured_evidence(plan, captures)

    assert measured["denominators"]["abstained_vial_decisions"] == 3


def test_preserves_nondefault_programmed_dispense_order(tmp_path):
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    paths = _captures(tmp_path)
    order = ["t100", "t200", "t400"]
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        by_ordinal = {
            ordinal: f"after-{tank}" for ordinal, tank in enumerate(order, start=1)
        }
        for slot in manifest["stage_slots"]:
            slot["stage_id"] = by_ordinal[slot["stage_ordinal"]]
        for row in manifest["rows"]:
            row["stage_id"] = by_ordinal[row["stage_ordinal"]]
        path.write_text(json.dumps(manifest), encoding="utf-8")
    context = _context()
    for batch in context["batches"]:
        batch["programmed_recipe"]["dispense_order"] = order

    plan = build_model_input_plan(paths, context)

    assert [
        stage["stage_id"] for stage in plan["decision_records"][0]["stages"]
    ] == ["after-t100", "after-t200", "after-t400"]
    assert plan["decision_records"][0]["programmed_dispense_order"] == order


def test_rejects_ambiguous_duplicate_recipe_representation(tmp_path):
    from fluidmix.explanation.vlm.model_input_plan_builder import ModelInputPlanError, build_model_input_plan

    context = _context()
    context["batches"][0]["programmed_dispense_order"] = ["t100", "t200", "t400"]
    context["batches"][0]["previous_programmed_ml"] = {
        "t100": 1,
        "t200": 1,
        "t400": 1,
    }
    context["batches"][0]["current_programmed_ml"] = {
        "t100": 1,
        "t200": 1,
        "t400": 1,
    }

    with pytest.raises(ModelInputPlanError, match="batch_context_batch_schema_invalid"):
        build_model_input_plan(_captures(tmp_path), context)


def test_public_cli_uses_strict_context_snapshot_and_refuses_clobber(tmp_path):
    paths = _captures(tmp_path)
    context_path = tmp_path / "batch-context.json"
    context_path.write_text(
        json.dumps(_context(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "model-input-plan.json"
    command = [sys.executable, str(SCRIPT)]
    for path in reversed(paths):
        command.extend(("--capture-manifest", str(path)))
    command.extend(("--batch-context", str(context_path), "--out", str(output)))

    first = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["source_batch_context_sha256"] == hashlib.sha256(
        context_path.read_bytes()
    ).hexdigest()
    assert list(plan["source_capture_manifest_sha256"]) == ["REF-A", "REF-B", "TARGET"]
    original = output.read_bytes()

    second = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert second.returncode != 0
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(f".{output.name}.tmp-*"))


def test_end_to_end_measured_evidence_and_no_clobber(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest, build_measured_evidence
    from fluidmix.explanation.vlm.model_input_plan_builder import write_model_input_plan
    from fluidmix.explanation.template.stage_cosmos_explanation import stage_reference_response

    paths = _captures(tmp_path)
    output = tmp_path / "model-input-plan.json"
    plan = write_model_input_plan(paths, _context(), output)
    captures = [
        bind_stage_capture_manifest(json.loads(path.read_text()), base_dir=path.parent / path.stem)
        for path in paths
    ]
    measured = build_measured_evidence(plan, captures)
    assert measured["artifact_type"] == "measured_evidence"
    assert len(measured["decision_records"]) == 3
    assert "Evidence:" in stage_reference_response(measured["decision_records"][0])
    with pytest.raises(FileExistsError):
        write_model_input_plan(paths, _context(), output)


def test_four_target_batches_feed_frozen_twelve_decision_cosmos_matrix(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        bind_stage_capture_manifest,
        build_measured_evidence,
    )
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan
    from fluidmix.explanation.template.stage_cosmos_explanation import build_stage_attempt_plan

    names = ("TARGET", "TARGET-2", "TARGET-3", "TARGET-4", "REF-A", "REF-B")
    paths = _captures(tmp_path, names)
    context = _context(names)
    for batch in context["batches"]:
        batch["cohort"] = (
            "scored_target"
            if batch["batch_id"].startswith("TARGET")
            else "reference_only"
        )
    plan = build_model_input_plan(paths, context)
    captures = [
        bind_stage_capture_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            base_dir=path.parent / path.stem,
        )
        for path in paths
    ]
    measured = build_measured_evidence(plan, captures)
    prompt = (ROOT / "data" / "cosmos_stage_prompt_v1.txt").read_text(
        encoding="utf-8"
    )

    attempts = build_stage_attempt_plan(measured, prompt)

    assert len(measured["decision_records"]) == 12
    primary_stages = [
        stage
        for decision in measured["decision_records"]
        for stage in decision["stages"]
    ]
    assert len(primary_stages) == 36
    assert all(stage["validity"] == "insufficient_evidence" for stage in primary_stages)
    assert all(
        {
            "target_rgb_bytes_reused_as_reference",
            "duplicate_reference_rgb_bytes",
        }.issubset(stage["limit_codes"])
        for stage in primary_stages
    )
    assert measured["denominators"] == {
        "image_measurements": 54,
        "valid_image_measurements": 54,
        "invalid_image_measurements": 0,
        "invalid_image_measurement_reasons": {},
        "vial_decisions": 12,
        "valid_vial_decisions": 0,
        "abstained_vial_decisions": 12,
    }
    assert len(attempts) == 48
    assert sum(row["kind"] == "model" for row in attempts) == 36
    assert sum(row["kind"] == "control" for row in attempts) == 12


def test_measurement_denominators_report_invalid_images_by_reason(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan
    from fluidmix.measurement.vision_measurement import evaluate_model_plan

    paths = _captures(tmp_path)
    plan = build_model_input_plan(paths, _context())
    captures = [
        bind_stage_capture_manifest(
            json.loads(path.read_text(encoding="utf-8")),
            base_dir=path.parent / path.stem,
        )
        for path in paths
    ]
    images = {
        image_id: dict(locator)
        for capture in captures
        for image_id, locator in capture["image_locators"]["images"].items()
    }
    first = images[sorted(images)[0]]
    mask_path = Path(first["mask_path"])
    with Image.open(mask_path) as image:
        size = image.size
    Image.fromarray(np.zeros((size[1], size[0]), dtype=np.uint8), mode="L").save(
        mask_path
    )
    first["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()

    measured = evaluate_model_plan(
        plan,
        {"schema_version": 1, "data_origin": "twin", "images": images},
    )

    assert measured["denominators"]["image_measurements"] == 27
    assert measured["denominators"]["valid_image_measurements"] == 26
    assert measured["denominators"]["invalid_image_measurements"] == 1
    assert measured["denominators"]["invalid_image_measurement_reasons"] == {
        "mask_empty": 1
    }

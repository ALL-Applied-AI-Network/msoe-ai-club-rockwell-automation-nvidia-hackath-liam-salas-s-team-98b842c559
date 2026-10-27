"""Tests for materializing Varied V3 stage-publication requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _model_plan(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "cohort"
    records: list[dict[str, object]] = []
    run_specs = (
        ("RUN-TRAIN-01", "train", "5", (10, 10, 10)),
        ("RUN-TRAIN-02", "train", "6", (30, 10, 0)),
        ("RUN-TRAIN-03", "train", "7", (20, 0, 50)),
        ("RUN-TRAIN-04", "train", "8", (20, 20, 20)),
        ("RUN-VALIDATION-01", "validation", "13", (0, 50, 20)),
        ("RUN-EVALUATION-01", "evaluation", "14", (60, 0, 10)),
    )
    fixed_order = ("t200", "t400", "t100")
    for run_number, (run_id, split, recipe_id, volumes) in enumerate(
        run_specs, start=1
    ):
        current = dict(zip(("t100", "t200", "t400"), volumes, strict=True))
        order = [tank for tank in fixed_order if current[tank] > 0]
        batch_id = f"BATCH-{run_number}"
        for position in (1, 2, 3):
            case_id = f"CASE-{run_number}-{position}"
            previous_recipe = str(int(recipe_id) - 1) if position == 1 else recipe_id
            previous = (
                {"t100": 10, "t200": 20, "t400": 30}
                if position == 1
                else current
            )
            records.append(
                {
                    "batch_id": batch_id,
                    "case_id": case_id,
                    "current_programmed_ml": current,
                    "current_recipe_id": recipe_id,
                    "data_origin": "twin",
                    "previous_programmed_ml": previous,
                    "previous_recipe_id": previous_recipe,
                    "programmed_dispense_order": order,
                    "quantitative_evidence_ids": [],
                    "run_id": run_id,
                    "split_role": split,
                    "split_version": "whole-run-varied-v3",
                    "stage_image_evidence_ids": [],
                    "vial_position": position,
                }
            )
            _write_json(
                root / "cases" / case_id / "app-output" / "receipt.json",
                {"case_id": case_id},
            )
    allowlist = sorted(records[0])
    plan = {
        "artifact_type": "synthetic_varied_run_input_plan_v3",
        "claim_boundary": "synthetic unrendered input placeholders only",
        "data_origin": "twin",
        "denominators": {"rendered_vials": 0, "runs": 6, "vials": 18},
        "model_input_allowlist": allowlist,
        "real_images_used": False,
        "schema_version": 3,
        "split_version": "whole-run-varied-v3",
        "status": "unrendered",
        "vial_records": records,
    }
    plan_path = tmp_path / "model-input-plan.json"
    _write_json(plan_path, plan)
    return plan_path, root


def _all_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [*value, *(key for child in value.values() for key in _all_keys(child))]
    if isinstance(value, list):
        return [key for child in value for key in _all_keys(child)]
    return []


def test_materializes_six_whole_run_requests_with_42_active_stage_pairs(tmp_path):
    from fluidmix.f2_varied_stage_materializer import (
        materialize_varied_publish_requests,
    )

    plan_path, cohort_root = _model_plan(tmp_path)
    output = tmp_path / "requests"

    receipt_path = materialize_varied_publish_requests(
        model_input_plan=plan_path,
        cohort_root=cohort_root,
        runtime_id="brev-isaac62-varied-v3",
        output_dir=output,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["batch_count"] == 6
    assert receipt["vial_receipt_count"] == 18
    assert receipt["stage_pair_count"] == 42
    assert receipt["split_vial_counts"] == {
        "evaluation": 3,
        "train": 12,
        "validation": 3,
    }
    requests = [
        json.loads((output / row["path"]).read_text(encoding="utf-8"))
        for row in receipt["requests"]
    ]
    assert {request["split_version"] for request in requests} == {
        "whole-run-varied-v3"
    }
    assert {len(request["programmed_recipe"]["dispense_order"]) for request in requests} == {
        2,
        3,
    }
    assert all(len(request["vials"]) == 3 for request in requests)
    assert any(
        len({vial["previous_recipe_id"] for vial in request["vials"]}) == 2
        for request in requests
    )
    forbidden = ("label", "residual", "palette", "expected_disposition")
    assert not any(
        fragment in key.lower()
        for request in requests
        for key in _all_keys(request)
        for fragment in forbidden
    )
    for request in requests:
        for vial in request["vials"]:
            source = Path(vial["receipt_path"])
            assert vial["receipt_sha256"] == _sha(source)

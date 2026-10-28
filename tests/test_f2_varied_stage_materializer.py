"""Tests for materializing Varied V3 stage-publication requests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


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
            case_number = (run_number - 1) * 3 + position
            case_id = f"CASE_{case_number:016x}"
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


def _model_plan_v4(tmp_path: Path) -> tuple[Path, Path]:
    from fluidmix.physx_varied_cohort_manifest_v4 import (
        build_varied_cohort_artifacts,
    )

    root = tmp_path / "cohort-v4"
    artifacts = build_varied_cohort_artifacts(repo_root=ROOT)
    plan = artifacts["model-input-plan.json"]
    render_plan = artifacts["render-schedule-plan.json"]
    records = plan["vial_records"]
    case_ids = [row["case_id"] for row in records]
    for case_id in case_ids:
        _write_json(
            root / "cases" / case_id / "app-output" / "receipt.json",
            {"case_id": case_id},
        )
    render_plan_sha = _canonical_sha256(render_plan)
    _write_json(
        root / "cohort-launch.json",
        {
            "schema_version": 1,
            "artifact_type": "physx_train_residual_serial_cohort_launch",
            "case_ids": case_ids,
            "plan_sha256": render_plan_sha,
            "expected_output_contract": {
                "case_count": 30,
                "cases": {case_id: {} for case_id in case_ids},
            },
        },
    )
    _write_json(
        root / "shard-merge.json",
        {
            "schema_version": 1,
            "artifact_type": "physx_train_residual_shard_merge",
            "status": "pass",
            "case_ids": case_ids,
            "merged_case_count": 30,
            "plan_sha256": render_plan_sha,
            "shard_count": 30,
            "source_shards": [
                {"index": index, "case_ids": [case_id]}
                for index, case_id in enumerate(case_ids)
            ],
            "full_resume_summary": {
                "status": "pass",
                "planned_case_count": 30,
                "validated_completed_case_count": 30,
                "plan_sha256": render_plan_sha,
                "case_results": [
                    {"case_id": case_id, "outcome": "recognized_completed"}
                    for case_id in case_ids
                ],
            },
        },
    )
    plan_path = tmp_path / "model-input-plan-v4.json"
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


def test_materializes_v4_contract_with_30_vials_and_source_bound_stage_pairs(tmp_path):
    from fluidmix.f2_varied_stage_materializer import materialize_varied_publish_requests

    plan_path, cohort_root = _model_plan_v4(tmp_path)
    output = tmp_path / "requests-v4"

    receipt_path = materialize_varied_publish_requests(
        model_input_plan=plan_path,
        cohort_root=cohort_root,
        runtime_id="brev-isaac62-varied-v4",
        output_dir=output,
    )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["contract_version"] == "whole-run-varied-v4"
    assert receipt["batch_count"] == 10
    assert receipt["vial_receipt_count"] == 30
    assert receipt["stage_pair_count"] == 78
    assert receipt["split_vial_counts"] == {
        "evaluation": 9,
        "train": 12,
        "validation": 9,
    }
    assert receipt["source_merge_receipt_sha256"] == _sha(
        cohort_root / "shard-merge.json"
    )
    assert receipt["source_cohort_launch_sha256"] == _sha(
        cohort_root / "cohort-launch.json"
    )
    assert receipt["training_eligibility"] == {
        "protocol": "inn-7-splits-v2",
        "split_spec_sha256": _sha(ROOT / "data/sim_split_spec.json"),
        "status": "ineligible",
        "training_handoff_allowed": False,
        "reason_codes": ["reserved_real_recipe_overlap"],
        "overlapping_vials": 30,
        "overlapping_runs": 10,
        "overlapping_current_recipe_ids": [
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
        ],
        "claim_boundary": (
            "renderer and engineering evidence only under inn-7-splits-v2; "
            "the reserved real recipes are not unseen"
        ),
    }
    requests = [
        json.loads((output / row["path"]).read_text(encoding="utf-8"))
        for row in receipt["requests"]
    ]
    assert {request["schema_version"] for request in requests} == {4}
    assert {request["split_version"] for request in requests} == {
        "whole-run-varied-v4"
    }


def test_v4_rejects_a_shape_valid_but_noncanonical_model_plan(tmp_path):
    from fluidmix.f2_varied_stage_materializer import (
        F2VariedStageMaterializationError,
        materialize_varied_publish_requests,
    )

    plan_path, cohort_root = _model_plan_v4(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["vial_records"][0]["previous_recipe_id"] = "13"
    _write_json(plan_path, plan)

    with pytest.raises(
        F2VariedStageMaterializationError,
        match="model_input_plan_source_binding_invalid",
    ):
        materialize_varied_publish_requests(
            model_input_plan=plan_path,
            cohort_root=cohort_root,
            runtime_id="brev-isaac62-varied-v4",
            output_dir=tmp_path / "noncanonical-v4-requests",
        )


def test_v4_requires_the_successful_exact_30_case_merge_receipt(tmp_path):
    from fluidmix.f2_varied_stage_materializer import (
        F2VariedStageMaterializationError,
        materialize_varied_publish_requests,
    )

    plan_path, cohort_root = _model_plan_v4(tmp_path)
    (cohort_root / "shard-merge.json").unlink()

    with pytest.raises(
        F2VariedStageMaterializationError,
        match="cohort_merge_binding_invalid",
    ):
        materialize_varied_publish_requests(
            model_input_plan=plan_path,
            cohort_root=cohort_root,
            runtime_id="brev-isaac62-varied-v4",
            output_dir=tmp_path / "missing-v4-merge-requests",
        )


def _rewrite_first_case_id(plan_path: Path, case_id: str) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["vial_records"][0]["case_id"] = case_id
    _write_json(plan_path, plan)


@pytest.mark.parametrize(
    "case_id",
    [
        "../outside/CASE_0000000000000001",
        "CASE_000000000000000g",
        "CASE_000000000000001",
    ],
)
def test_rejects_noncanonical_or_traversal_case_ids(tmp_path, case_id):
    from fluidmix.f2_varied_stage_materializer import (
        F2VariedStageMaterializationError,
        materialize_varied_publish_requests,
    )

    plan_path, cohort_root = _model_plan(tmp_path)
    _rewrite_first_case_id(plan_path, case_id)

    with pytest.raises(F2VariedStageMaterializationError, match="case_id_invalid"):
        materialize_varied_publish_requests(
            model_input_plan=plan_path,
            cohort_root=cohort_root,
            runtime_id="brev-isaac62-varied-v3",
            output_dir=tmp_path / "invalid-case-id-requests",
        )


def test_rejects_absolute_case_id_even_when_outside_receipt_exists(tmp_path):
    from fluidmix.f2_varied_stage_materializer import (
        F2VariedStageMaterializationError,
        materialize_varied_publish_requests,
    )

    plan_path, cohort_root = _model_plan(tmp_path)
    outside_case = (tmp_path / "outside-case").resolve()
    _write_json(outside_case / "app-output" / "receipt.json", {"outside": True})
    _rewrite_first_case_id(plan_path, str(outside_case))

    with pytest.raises(F2VariedStageMaterializationError, match="case_id_invalid"):
        materialize_varied_publish_requests(
            model_input_plan=plan_path,
            cohort_root=cohort_root,
            runtime_id="brev-isaac62-varied-v3",
            output_dir=tmp_path / "absolute-case-id-requests",
        )


def test_rejects_case_directory_symlink_escape(tmp_path):
    from fluidmix.f2_varied_stage_materializer import (
        F2VariedStageMaterializationError,
        materialize_varied_publish_requests,
    )

    plan_path, cohort_root = _model_plan(tmp_path)
    case_id = "CASE_0000000000000001"
    case_root = cohort_root / "cases" / case_id
    outside = tmp_path / "outside-symlink-target"
    shutil.move(case_root, outside)
    case_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        F2VariedStageMaterializationError,
        match=f"case_receipt_invalid:{case_id}",
    ):
        materialize_varied_publish_requests(
            model_input_plan=plan_path,
            cohort_root=cohort_root,
            runtime_id="brev-isaac62-varied-v3",
            output_dir=tmp_path / "symlink-case-requests",
        )

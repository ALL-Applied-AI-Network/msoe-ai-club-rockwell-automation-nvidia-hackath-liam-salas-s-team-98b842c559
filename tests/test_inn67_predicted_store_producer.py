from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.inn67_predicted_store_producer import (  # noqa: E402
    CANDIDATE_SHA256,
    CHECKPOINT_SHA256,
    REPLAY_SHA256,
    REVIEW_SHA256,
    PredictedStoreProducerError,
    produce_predicted_store,
)
import fluidmix.inn67_predicted_store_producer as producer_module  # noqa: E402
from test_inn67_bank_materializer import _fixture  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_unavailable_lineage_bytes(monkeypatch):
    monkeypatch.setattr(producer_module, "_verify_lineage_file", lambda _path, _expected: True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_fixture(tmp_path: Path):
    plan_root, amendment, materialization_amendment, inventory, _ = _fixture(tmp_path)
    plan = json.loads((plan_root / "reference-bank-plan.json").read_text())
    inventory_value = json.loads(inventory.read_text())
    by_renderer = {
        row["renderer_case_id"]: row
        for subrun in inventory_value["subruns"]
        for row in subrun["cases"]
    }
    lineage = {}
    for role, digest in {
        "checkpoint": CHECKPOINT_SHA256,
        "candidate": CANDIDATE_SHA256,
        "replay": REPLAY_SHA256,
        "review": REVIEW_SHA256,
    }.items():
        path = tmp_path / "lineage" / role
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        # A private verifier seam is stubbed because production lineage bytes are unavailable.
        lineage[role] = {"path": str(path.resolve()), "sha256": digest}

    cases = []
    for case_index, case in enumerate(plan["cases"]):
        image_root = tmp_path / "images" / case["case_id"]
        image_root.mkdir(parents=True, exist_ok=True)
        prefill = tmp_path / "images" / "prefill-by-run" / f"{case['run_id']}.png"
        prefill.parent.mkdir(parents=True, exist_ok=True)
        if not prefill.exists():
            Image.new("RGB", (8, 6), color=(5, 7, 11)).save(prefill)
        stages = []
        for stage in case["stages"]:
            rgb = image_root / f"stage-{stage['ordinal']}.png"
            image = Image.new(
                "RGB", (8, 6),
                color=(30 + int(case["recipe_id"][-2:]), 40 + stage["ordinal"], 50),
            )
            image.putpixel((0, 0), (case_index, stage["ordinal"], 1))
            image.save(rgb)
            stages.append({
                "stage_ordinal": stage["ordinal"],
                "rgb_path": str(rgb.resolve()),
                "rgb_sha256": _sha(rgb),
            })
        inventory_case = by_renderer[case["renderer_case_id"]]
        cases.append({
            "case_id": case["case_id"],
            "run_id": case["run_id"],
            "batch_id": case["batch_id"],
            "recipe_id": case["recipe_id"],
            "render_receipt_sha256": inventory_case["receipt_sha256"],
            "validation_receipt_sha256": inventory_case["validation_receipt_sha256"],
            "prefill_rgb_path": str(prefill.resolve()),
            "prefill_rgb_sha256": _sha(prefill),
            "stages": stages,
        })
    source = {
        "schema_version": 1,
        "artifact_type": "inn67_prediction_source_manifest",
        "status": "accepted",
        "render_inventory_sha256": _sha(inventory),
        "reference_bank_plan_sha256": _sha(plan_root / "reference-bank-plan.json"),
        "known_good_profile_sha256": _sha(plan_root / "known-good-profile.json"),
        "lineage": lineage,
        "cases": cases,
    }
    source_path = tmp_path / "source.json"
    _write_json(source_path, source)
    return plan_root, amendment, materialization_amendment, inventory, source_path


def _predict(current: Path, reference: Path) -> np.ndarray:
    assert current != reference
    with Image.open(current) as image:
        width, height = image.size
    mask = np.zeros((height, width), dtype=bool)
    mask[1:, 1:-1] = True
    return mask


def test_produces_exact_48_row_store_export_in_donor_first_plan_order(tmp_path: Path):
    plan_root, amendment, materialization_amendment, inventory, source = _source_fixture(tmp_path)
    output = tmp_path / "output"
    result = produce_predicted_store(
        plan_root=plan_root,
        amendment_path=amendment,
        materialization_amendment_path=materialization_amendment,
        render_inventory_path=inventory,
        source_manifest_path=source,
        output_dir=output,
        predictor=_predict,
    )

    export = json.loads((output / "predicted-store-export.json").read_text())
    receipt = json.loads((output / "population-receipt.json").read_text())
    assert result["record_count"] == 48
    assert len(export["records"]) == 48
    assert [row["event_cursor"] for row in export["records"]] == list(range(48))
    assert all(row["batch_id"] != "INN67-DEV-RB-B03" for row in export["records"][:32])
    assert all(row["batch_id"] == "INN67-DEV-RB-B03" for row in export["records"][32:])
    assert receipt["artifact_type"] == "inn67_predicted_evidence_store_population"
    assert receipt["records_sha256"]
    assert receipt["lineage"] == {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "replay_sha256": REPLAY_SHA256,
        "review_sha256": REVIEW_SHA256,
    }
    with sqlite3.connect(output / "evidence.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM predicted_records").fetchone() == (48,)

    from fluidmix.inn67_bank_materializer import materialize_reference_bank
    materialized = materialize_reference_bank(
        plan_root=plan_root,
        amendment_path=amendment,
        materialization_amendment_path=materialization_amendment,
        render_inventory_path=inventory,
        predicted_store_export_path=output / "predicted-store-export.json",
        output_dir=tmp_path / "bank",
    )
    assert materialized["status"] == "published"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda value: value["cases"].pop(), "source_case_denominator_invalid"),
        (lambda value: value["cases"].append(deepcopy(value["cases"][0])), "source_case_denominator_invalid"),
        (lambda value: value["cases"][0].update({"batch_id": "INN67-DEV-RB-B03"}), "source_plan_binding_mismatch"),
        (lambda value: value["cases"][0].update({"split_role": "test"}), "forbidden_test_or_video5_locator"),
        (lambda value: value["cases"][0].update({"prefill_rgb_path": "/tmp/video5.png"}), "forbidden_test_or_video5_locator"),
        (lambda value: value["cases"][0]["stages"].pop(), "source_stage_denominator_invalid"),
        (lambda value: value["cases"][0]["stages"].__setitem__(1, deepcopy(value["cases"][0]["stages"][0])), "source_stage_plan_binding_mismatch"),
        (lambda value: value["cases"][1].update({"prefill_rgb_path": value["cases"][6]["prefill_rgb_path"], "prefill_rgb_sha256": value["cases"][6]["prefill_rgb_sha256"]}), "same_run_prefill_drift"),
        (lambda value: value["cases"][1]["stages"][0].update({"rgb_path": value["cases"][0]["stages"][0]["rgb_path"], "rgb_sha256": value["cases"][0]["stages"][0]["rgb_sha256"]}), "stage_rgb_duplicate"),
        (lambda value: value["cases"][0]["stages"][0].update({"rgb_sha256": "0" * 64}), "stage_rgb_hash_mismatch"),
        (lambda value: value["lineage"]["replay"].update({"sha256": "0" * 64}), "lineage_identity_mismatch"),
    ],
)
def test_source_contract_fails_closed(tmp_path: Path, mutation, error: str):
    plan_root, amendment, materialization_amendment, inventory, source = _source_fixture(tmp_path)
    value = json.loads(source.read_text())
    mutation(value)
    _write_json(source, value)
    with pytest.raises(PredictedStoreProducerError, match=error):
        produce_predicted_store(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=inventory,
            source_manifest_path=source,
            output_dir=tmp_path / "output",
            predictor=_predict,
        )


def test_no_clobber_fails_closed(tmp_path: Path):
    plan_root, amendment, materialization_amendment, inventory, source = _source_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(PredictedStoreProducerError, match="output_exists"):
        produce_predicted_store(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=inventory,
            source_manifest_path=source,
            output_dir=output,
            predictor=_predict,
        )


def test_nonfinite_predictor_output_fails_closed_without_partial_output(tmp_path: Path):
    plan_root, amendment, materialization_amendment, inventory, source = _source_fixture(tmp_path)
    output = tmp_path / "output"

    def nonfinite(current: Path, reference: Path) -> np.ndarray:
        del current, reference
        return np.full((6, 8), np.nan)

    with pytest.raises(PredictedStoreProducerError, match="predicted_mask_invalid"):
        produce_predicted_store(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=inventory,
            source_manifest_path=source,
            output_dir=output,
            predictor=nonfinite,
        )
    assert not output.exists()

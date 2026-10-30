from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.inn67_predicted_store_producer import (  # noqa: E402
    CANDIDATE_RECEIPT_SHA256,
    CHECKPOINT_SHA256,
    CNN_PREDICTOR_SOURCE_SHA256,
    EMPTY_REFERENCE_SOURCE_SHA256,
    REPLAY_SHA256,
    REVIEW_SHA256,
    PredictedStoreProducerError,
    produce_predicted_store,
)
from fluidmix.inn67_postrender_producer import produce_postrender_inputs  # noqa: E402
import fluidmix.inn67_predicted_store_producer as producer_module  # noqa: E402
from test_inn67_bank_materializer import _fixture  # noqa: E402
from test_inn67_postrender_producer import (  # noqa: E402
    _fixture as _postrender_fixture,
    _passing_validator,
)


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
    lineage_root = tmp_path / "lineage"
    checkpoint = lineage_root / "checkpoint.pt"
    candidate = lineage_root / "candidate-receipt.json"
    replay = lineage_root / "replay.json"
    review = lineage_root / "review.md"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic-checkpoint")
    _write_json(candidate, {
        "artifact_type": "wp5_cnn_frozen_candidate",
        "source_commit": "a5f29ea90272a3424d8c7de09dbf435509d28105",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "threshold": 0.5,
    })
    review.write_text("# Independent synthetic fixture\n")
    _write_json(replay, {
        "artifact_type": "wp5_cnn_deterministic_bounds_replay",
        "status": "swap_review_ready",
        "checkpoint_frozen": True,
        "threshold_frozen": 0.5,
        "scoring": {
            "all": {"all_gates_pass": True, "decision": "replace_approach_a"},
            "cleaned": {"all_gates_pass": True, "decision": "replace_approach_a"},
        },
        "source_bindings": {
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": CHECKPOINT_SHA256},
            "candidate_receipt": {"path": str(candidate.resolve()), "sha256": CANDIDATE_RECEIPT_SHA256},
        },
    })
    lineage = {"checkpoint": checkpoint, "replay": replay, "review": review, "candidate": candidate}

    for case_index, case in enumerate(plan["cases"]):
        inventory_case = by_renderer[case["renderer_case_id"]]
        receipt_path = Path(inventory_case["receipt_path"])
        image_root = receipt_path.parent / "rgb"
        image_root.mkdir(parents=True, exist_ok=True)
        prefill = image_root / "prefill.png"
        Image.new("RGB", (640, 480), color=(5, 7, 11)).save(prefill)
        pairs = []
        for stage in case["stages"]:
            rgb = image_root / f"stage-{stage['ordinal']}.png"
            image = Image.new(
                "RGB", (640, 480),
                color=(30 + int(case["recipe_id"][-2:]), 40 + stage["ordinal"], 50),
            )
            image.putpixel((0, 0), (case_index, stage["ordinal"], 1))
            image.save(rgb)
            pairs.append({
                "phase": f"{stage['tank_id']}-stage",
                "rgb_path": str(rgb.relative_to(receipt_path.parent)),
                "rgb_sha256": _sha(rgb),
            })
        receipt = json.loads(receipt_path.read_text())
        receipt.update({
            "pre_pour_parity": {
                "f2_prefill_path": str(prefill.relative_to(receipt_path.parent)),
                "f2_prefill_sha256": _sha(prefill),
            },
            "semantics": {"pairs": pairs},
        })
        _write_json(receipt_path, receipt)
        inventory_case["receipt_sha256"] = _sha(receipt_path)
        inventory_case["prefill_rgb_path"] = str(prefill)
        inventory_case["prefill_rgb_sha256"] = _sha(prefill)
        inventory_case["stage_rgbs"] = [
            {
                "stage_ordinal": stage["ordinal"],
                "tank_id": stage["tank_id"],
                "rgb_path": str(image_root / f"stage-{stage['ordinal']}.png"),
                "rgb_sha256": pair["rgb_sha256"],
            }
            for stage, pair in zip(case["stages"], pairs, strict=True)
        ]
        validation_path = Path(inventory_case["validation_receipt_path"])
        validation = json.loads(validation_path.read_text())
        validation["source_render_receipt_sha256"] = inventory_case["receipt_sha256"]
        _write_json(validation_path, validation)
        inventory_case["validation_receipt_sha256"] = _sha(validation_path)
    _write_json(inventory, inventory_value)
    return plan_root, amendment, materialization_amendment, inventory, lineage


def _predict(current: Path, reference: Path) -> np.ndarray:
    assert current != reference
    with Image.open(current) as image:
        width, height = image.size
    mask = np.zeros((height, width), dtype=bool)
    mask[1:, 1:-1] = True
    return mask


def _produce(fixture, output: Path, predictor=_predict):
    plan_root, amendment, materialization_amendment, inventory, lineage = fixture
    with patch.object(
        producer_module, "_load_default_predictor", return_value=predictor
    ):
        return produce_predicted_store(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=inventory,
            cnn_checkpoint_path=lineage["checkpoint"],
            cnn_replay_path=lineage["replay"],
            cnn_review_path=lineage["review"],
            output_dir=output,
        )


def test_produces_exact_48_row_store_export_in_donor_first_plan_order(tmp_path: Path):
    fixture = _source_fixture(tmp_path)
    plan_root, amendment, materialization_amendment, inventory, _ = fixture
    output = tmp_path / "output"
    result = _produce(fixture, output)

    export = json.loads((output / "predicted-store-export.json").read_text())
    receipt = json.loads((output / "population-receipt.json").read_text())
    derived = json.loads((output / "derived-source-manifest.json").read_text())
    assert result["record_count"] == 48
    assert len(export["records"]) == 48
    assert [row["event_cursor"] for row in export["records"]] == list(range(48))
    assert all(row["batch_id"] != "INN67-DEV-RB-B03" for row in export["records"][:32])
    assert all(row["batch_id"] == "INN67-DEV-RB-B03" for row in export["records"][32:])
    assert receipt["artifact_type"] == "inn67_predicted_evidence_store_population"
    assert receipt["records_sha256"]
    assert derived["case_count"] == 18
    assert derived["stage_count"] == 48
    assert sum(len(case["stages"]) for case in derived["cases"]) == 48
    assert receipt["source_bindings"]["derived_source_manifest_sha256"] == _sha(
        output / "derived-source-manifest.json"
    )
    assert receipt["lineage"] == {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "candidate_receipt_sha256": CANDIDATE_RECEIPT_SHA256,
        "deterministic_replay_sha256": REPLAY_SHA256,
        "independent_review_sha256": REVIEW_SHA256,
        "predictor_source_sha256": CNN_PREDICTOR_SOURCE_SHA256,
        "empty_reference_source_sha256": EMPTY_REFERENCE_SOURCE_SHA256,
        "config_sha256": export["predictor"]["config_sha256"],
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

    from fluidmix.wp7_development_reference import (
        DevelopmentReferenceError,
        materialize_development_reference_store,
        validate_development_reference_inputs,
    )

    assessment = validate_development_reference_inputs(
        predicted_store_export_path=output / "predicted-store-export.json",
        development_reference_bank_path=tmp_path / "bank" / "development-reference-bank.json",
        delta_e_frozen_config_path=tmp_path / "bank" / "delta-e-frozen-config.json",
        materialization_receipt_path=tmp_path / "bank" / "materialization-receipt.json",
    )
    assert assessment["status"] == "blocked_missing_normal_evidence_fields"
    assert assessment["donor_record_count"] == 32
    assert {"fill_clip_path", "rgb_path", "mask_path", "mad_lab", "recipe_ml"} <= set(
        assessment["missing_normal_evidence_fields"]
    )
    derived_output = tmp_path / "development-reference-store"
    with pytest.raises(DevelopmentReferenceError, match="normal_evidence_store_fields_missing"):
        materialize_development_reference_store(
            predicted_store_export_path=output / "predicted-store-export.json",
            development_reference_bank_path=tmp_path / "bank" / "development-reference-bank.json",
            delta_e_frozen_config_path=tmp_path / "bank" / "delta-e-frozen-config.json",
            materialization_receipt_path=tmp_path / "bank" / "materialization-receipt.json",
            output_dir=derived_output,
        )
    assert not derived_output.exists()

    inputs = {
        "predicted_store_export_path": output / "predicted-store-export.json",
        "development_reference_bank_path": tmp_path / "bank" / "development-reference-bank.json",
        "delta_e_frozen_config_path": tmp_path / "bank" / "delta-e-frozen-config.json",
        "materialization_receipt_path": tmp_path / "bank" / "materialization-receipt.json",
    }
    for role, source in (
        ("development_reference_bank_path", inputs["development_reference_bank_path"]),
        ("delta_e_frozen_config_path", inputs["delta_e_frozen_config_path"]),
        ("materialization_receipt_path", inputs["materialization_receipt_path"]),
        ("predicted_store_export_path", inputs["predicted_store_export_path"]),
    ):
        stale = json.loads(Path(source).read_text())
        stale["stale_hash_probe"] = True
        stale_path = tmp_path / f"stale-{Path(source).name}"
        _write_json(stale_path, stale)
        with pytest.raises(
            DevelopmentReferenceError,
            match="binding|contract|fields|cross",
        ):
            validate_development_reference_inputs(**{**inputs, role: stale_path})

    bank_value = json.loads(Path(inputs["development_reference_bank_path"]).read_text())
    receipt_value = json.loads(Path(inputs["materialization_receipt_path"]).read_text())
    duplicate_bank = deepcopy(bank_value)
    duplicate_bank["records"][-1] = deepcopy(duplicate_bank["records"][0])
    duplicate_path = tmp_path / "duplicate-bank.json"
    _write_json(duplicate_path, duplicate_bank)
    rebound_receipt = deepcopy(receipt_value)
    rebound_receipt["outputs"]["development-reference-bank.json"] = _sha(duplicate_path)
    rebound_receipt_path = tmp_path / "duplicate-bank-receipt.json"
    _write_json(rebound_receipt_path, rebound_receipt)
    with pytest.raises(DevelopmentReferenceError, match="donor_record_binding_mismatch"):
        validate_development_reference_inputs(
            **{
                **inputs,
                "development_reference_bank_path": duplicate_path,
                "materialization_receipt_path": rebound_receipt_path,
            }
        )

    missing_bank = deepcopy(bank_value)
    missing_bank["records"].pop()
    missing_path = tmp_path / "missing-bank.json"
    _write_json(missing_path, missing_bank)
    missing_receipt = deepcopy(receipt_value)
    missing_receipt["outputs"]["development-reference-bank.json"] = _sha(missing_path)
    missing_receipt_path = tmp_path / "missing-bank-receipt.json"
    _write_json(missing_receipt_path, missing_receipt)
    with pytest.raises(DevelopmentReferenceError, match="materialization_cross_binding_mismatch"):
        validate_development_reference_inputs(
            **{
                **inputs,
                "development_reference_bank_path": missing_path,
                "materialization_receipt_path": missing_receipt_path,
            }
        )

    export_value = json.loads(Path(inputs["predicted_store_export_path"]).read_text())
    population_value = json.loads((output / "population-receipt.json").read_text())
    stale_population = deepcopy(population_value)
    stale_population["record_count"] = 47
    stale_population_path = tmp_path / "stale-population.json"
    _write_json(stale_population_path, stale_population)
    population_rebound_export = deepcopy(export_value)
    population_rebound_export["store"].update(
        {
            "population_receipt_path": str(stale_population_path),
            "population_receipt_sha256": _sha(stale_population_path),
        }
    )
    population_rebound_export_path = tmp_path / "population-rebound-export.json"
    _write_json(population_rebound_export_path, population_rebound_export)
    with pytest.raises(DevelopmentReferenceError, match="population_export_binding_mismatch"):
        validate_development_reference_inputs(
            **{
                **inputs,
                "predicted_store_export_path": population_rebound_export_path,
            }
        )

    stale_database = tmp_path / "stale-evidence.sqlite3"
    shutil.copyfile(output / "evidence.sqlite3", stale_database)
    with stale_database.open("ab") as stream:
        stream.write(b"stale")
    stale_db_population = deepcopy(population_value)
    stale_db_population["store_sha256"] = _sha(stale_database)
    stale_db_population_path = tmp_path / "stale-db-population.json"
    _write_json(stale_db_population_path, stale_db_population)
    stale_db_export = deepcopy(export_value)
    stale_db_export["store"].update(
        {
            "database_path": str(stale_database),
            "database_sha256": _sha(stale_database),
            "population_receipt_path": str(stale_db_population_path),
            "population_receipt_sha256": _sha(stale_db_population_path),
        }
    )
    stale_db_export_path = tmp_path / "stale-db-export.json"
    _write_json(stale_db_export_path, stale_db_export)
    with pytest.raises(DevelopmentReferenceError, match="binding|mismatch"):
        validate_development_reference_inputs(
            **{**inputs, "predicted_store_export_path": stale_db_export_path}
        )

    occupied = tmp_path / "occupied-development-store"
    occupied.mkdir()
    (occupied / "owner.txt").write_text("do-not-remove")
    with pytest.raises(DevelopmentReferenceError, match="output_directory_exists"):
        materialize_development_reference_store(**inputs, output_dir=occupied)
    assert (occupied / "owner.txt").read_text() == "do-not-remove"


def test_postrender_inventory_flows_through_predicted_store_to_materializer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, amendment, materialization_amendment, _, lineage = _source_fixture(
        tmp_path / "support"
    )
    plan_root, render_root, _ = _postrender_fixture(tmp_path / "postrender")
    monkeypatch.setattr(
        "fluidmix.inn67_postrender_producer._load_canonical_validator",
        lambda _path: _passing_validator,
    )
    postrender_output = render_root / "accepted"
    postrender = produce_postrender_inputs(
        plan_root=plan_root,
        render_root=render_root,
        output_dir=postrender_output,
    )

    predicted_output = tmp_path / "predicted"
    with patch.object(
        producer_module, "_load_default_predictor", return_value=_predict
    ):
        predicted = produce_predicted_store(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=postrender_output / "render-inventory.json",
            cnn_checkpoint_path=lineage["checkpoint"],
            cnn_replay_path=lineage["replay"],
            cnn_review_path=lineage["review"],
            output_dir=predicted_output,
        )

    from fluidmix.inn67_bank_materializer import materialize_reference_bank

    materialized = materialize_reference_bank(
        plan_root=plan_root,
        amendment_path=amendment,
        materialization_amendment_path=materialization_amendment,
        render_inventory_path=postrender_output / "render-inventory.json",
        predicted_store_export_path=predicted_output / "predicted-store-export.json",
        output_dir=tmp_path / "bank",
    )
    assert postrender["case_count"] == 18
    assert predicted["record_count"] == 48
    assert materialized["status"] == "published"


def _mutate_first_receipt(fixture, mutation) -> None:
    inventory = fixture[3]
    value = json.loads(inventory.read_text())
    row = value["subruns"][0]["cases"][0]
    receipt_path = Path(row["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    mutation(receipt)
    _write_json(receipt_path, receipt)
    row["receipt_sha256"] = _sha(receipt_path)
    validation_path = Path(row["validation_receipt_path"])
    validation = json.loads(validation_path.read_text())
    validation["source_render_receipt_sha256"] = row["receipt_sha256"]
    _write_json(validation_path, validation)
    row["validation_receipt_sha256"] = _sha(validation_path)
    _write_json(inventory, value)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda receipt: receipt["semantics"]["pairs"].pop(), "case_rgb_declarations_invalid"),
        (lambda receipt: receipt["semantics"]["pairs"].append(deepcopy(receipt["semantics"]["pairs"][0])), "case_rgb_declarations_invalid"),
        (lambda receipt: receipt["semantics"]["pairs"].reverse(), "case_rgb_declarations_invalid"),
        (lambda receipt: receipt["arguments"].update({"train_case_id": "WRONG"}), "case_receipt_not_accepted"),
        (lambda receipt: receipt["semantics"]["pairs"][0].update({"rgb_sha256": "0" * 64}), "stage_rgb_binding_mismatch"),
        (lambda receipt: receipt["semantics"]["pairs"][0].update({"rgb_path": "masks/oracle.png"}), "stage_rgb_path_invalid"),
        (lambda receipt: receipt["pre_pour_parity"].update({"f2_prefill_path": "wp1/prefill.png"}), "prefill_rgb_path_invalid"),
    ],
)
def test_receipt_derived_source_contract_fails_closed(tmp_path: Path, mutation, error: str):
    fixture = _source_fixture(tmp_path)
    _mutate_first_receipt(fixture, mutation)
    with pytest.raises(PredictedStoreProducerError, match=error):
        _produce(fixture, tmp_path / "output")


def test_candidate_receipt_role_and_identity_fail_closed(tmp_path: Path):
    fixture = _source_fixture(tmp_path)
    lineage = fixture[4]
    replay = json.loads(lineage["replay"].read_text())
    replay["source_bindings"]["candidate_receipt"]["sha256"] = "0" * 64
    _write_json(lineage["replay"], replay)
    with pytest.raises(PredictedStoreProducerError, match="lineage_identity_mismatch"):
        _produce(fixture, tmp_path / "wrong-role")

    fixture = _source_fixture(tmp_path / "second")
    lineage = fixture[4]
    candidate = json.loads(lineage["candidate"].read_text())
    candidate["artifact_type"] = "comparison"
    _write_json(lineage["candidate"], candidate)
    with pytest.raises(PredictedStoreProducerError, match="candidate_receipt_invalid"):
        _produce(fixture, tmp_path / "wrong-candidate")


def test_render_receipt_mutation_without_inventory_rebind_fails_closed(tmp_path: Path):
    fixture = _source_fixture(tmp_path)
    inventory = json.loads(fixture[3].read_text())
    receipt_path = Path(inventory["subruns"][0]["cases"][0]["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["semantics"]["pairs"].pop()
    _write_json(receipt_path, receipt)
    with pytest.raises(PredictedStoreProducerError, match="render_receipt_hash_mismatch"):
        _produce(fixture, tmp_path / "output")


def test_no_clobber_fails_closed(tmp_path: Path):
    fixture = _source_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(PredictedStoreProducerError, match="output_exists"):
        _produce(fixture, output)


def test_nonfinite_predictor_output_fails_closed_without_partial_output(tmp_path: Path):
    fixture = _source_fixture(tmp_path)
    output = tmp_path / "output"

    def nonfinite(current: Path, reference: Path) -> np.ndarray:
        del current, reference
        return np.full((480, 640), np.nan)

    with pytest.raises(PredictedStoreProducerError, match="predicted_mask_invalid"):
        _produce(fixture, output, predictor=nonfinite)
    assert not output.exists()


def test_public_api_rejects_arbitrary_predictor_injection_before_publication(
    tmp_path: Path,
):
    fixture = _source_fixture(tmp_path)
    plan_root, amendment, materialization_amendment, inventory, lineage = fixture
    output = tmp_path / "output"

    with pytest.raises(TypeError, match="predictor"):
        produce_predicted_store(
            plan_root=plan_root,
            amendment_path=amendment,
            materialization_amendment_path=materialization_amendment,
            render_inventory_path=inventory,
            cnn_checkpoint_path=lineage["checkpoint"],
            cnn_replay_path=lineage["replay"],
            cnn_review_path=lineage["review"],
            output_dir=output,
            predictor=_predict,
        )

    assert not output.exists()
    assert not (tmp_path / "bank").exists()


def test_public_api_uses_frozen_loader_and_serializes_exact_accepted_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _source_fixture(tmp_path)
    plan_root, amendment, materialization_amendment, inventory, lineage = fixture
    loaded = []

    def load(checkpoint: Path):
        loaded.append(checkpoint)
        return _predict

    monkeypatch.setattr(producer_module, "_load_default_predictor", load)
    output = tmp_path / "output"
    produce_predicted_store(
        plan_root=plan_root,
        amendment_path=amendment,
        materialization_amendment_path=materialization_amendment,
        render_inventory_path=inventory,
        cnn_checkpoint_path=lineage["checkpoint"],
        cnn_replay_path=lineage["replay"],
        cnn_review_path=lineage["review"],
        output_dir=output,
    )

    export = json.loads((output / "predicted-store-export.json").read_text())
    assert loaded == [lineage["checkpoint"]]
    assert export["predictor"] == producer_module.ACCEPTED_CNN_PREDICTOR


def test_reviewed_cnn_runtime_source_is_present_and_exact():
    import fluidmix.cnn_predictor as runtime

    assert _sha(Path(runtime.__file__)) == CNN_PREDICTOR_SOURCE_SHA256
    assert runtime.ResNet18UNet(weights=None).__class__.__name__ == "ResNet18UNet"


def test_default_loader_loads_reviewed_runtime_with_cpu_checkpoint(tmp_path: Path):
    import torch
    from fluidmix.cnn_predictor import ResNet18UNet

    checkpoint = tmp_path / "synthetic.pt"
    torch.save(ResNet18UNet(weights=None).state_dict(), checkpoint)
    loaded = producer_module._load_default_predictor(checkpoint)
    assert callable(loaded)

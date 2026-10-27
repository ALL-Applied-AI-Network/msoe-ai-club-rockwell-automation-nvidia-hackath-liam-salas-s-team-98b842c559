"""Strict RGB-only predicted-mask locator handoff tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(root: Path, image_id: str, rgb_value=(0, 0, 0)) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    rgb_path = (root / f"{image_id}-rgb.png").resolve()
    mask_path = (root / f"{image_id}-mask.png").resolve()
    rgb = np.empty((3, 4, 3), dtype=np.uint8)
    rgb[:] = rgb_value
    rgb[2, 3] = tuple(hashlib.sha256(image_id.encode()).digest()[:3])
    mask = np.array(
        [[0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    return {
        "rgb_path": str(rgb_path),
        "rgb_sha256": _sha(rgb_path),
        "mask_path": str(mask_path),
        "mask_sha256": _sha(mask_path),
    }


def _stage(ordinal: int, target: str, peers: list[str]) -> dict[str, object]:
    return {
        "evidence_id": f"EVID-{target}",
        "measurement_id": f"MEAS-{target}",
        "stage_id": f"STAGE{ordinal}",
        "stage_ordinal": ordinal,
        "time_code": (171, 342, 686)[ordinal - 1],
        "target_image_id": target,
        "reference_cases": [
            {
                "case_id": "all_reference_batches",
                "mode": "all_reference_batches",
                "alternative_id": None,
                "candidate_peers": [
                    {
                        "peer_image_id": peer,
                        "batch_id": f"REF-{index}",
                        "run_id": f"RUN-REF-{index}",
                        "origin": "reference_bank",
                    }
                    for index, peer in enumerate(peers, 1)
                ],
                "disposition_source": True,
            }
        ],
        "primary_mode": "all_reference_batches",
        "unit": "delta_e76",
    }


def _plan() -> dict[str, object]:
    config = {
        "tight_cluster_max_diameter_delta_e76": 2.0,
        "threshold_floor_delta_e76": 2.0,
        "threshold_margin_delta_e76": 1.0,
    }
    return {
        "schema_version": 2,
        "artifact_type": "model_input_plan",
        "config_id": "predicted-mask-handoff-test-v1",
        "decision_config": config,
        "decision_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "decision_records": [
            {
                "decision_id": "DEC-BATCH-V1",
                "family_id": "TR",
                "batch_id": "BATCH_A1B2C3D4E5F6",
                "run_id": "RUN-TARGET",
                "cohort": "scored_target",
                "partition": "train",
                "previous_sim_recipe_id": "S0437",
                "current_sim_recipe_id": "S0117",
                "previous_programmed_ml": {"t100": 60, "t200": 10, "t400": 10},
                "current_programmed_ml": {"t100": 10, "t200": 40, "t400": 25},
                "programmed_dispense_order": ["t200", "t400", "t100"],
                "vial_id": "V1",
                "vial_position": 1,
                "stages": [
                    _stage(
                        ordinal,
                        f"TARGET-STAGE{ordinal}",
                        [f"PEER1-STAGE{ordinal}", f"PEER2-STAGE{ordinal}"],
                    )
                    for ordinal in range(1, 4)
                ],
                "decision_timing": (
                    "retrospective_after_complete_three_vial_batch_no_latency_claim"
                ),
            }
        ],
        "real_data_used": False,
    }


def _legacy_locators(tmp_path: Path) -> dict[str, object]:
    images = {}
    for ordinal in range(1, 4):
        images[f"TARGET-STAGE{ordinal}"] = _image(
            tmp_path, f"TARGET-STAGE{ordinal}"
        )
        for peer in (1, 2):
            image_id = f"PEER{peer}-STAGE{ordinal}"
            images[image_id] = _image(tmp_path, image_id)
    return {"schema_version": 1, "data_origin": "twin", "images": images}


def _producer() -> dict[str, object]:
    return {
        "producer_id": "rgb-segmenter-predictions-20261027",
        "producer_kind": "rgb_only_liquid_segmenter",
        "model_id": "mobilenet-unet-v1",
        "model_sha256": "a" * 64,
        "config_id": "mobilenet-unet-threshold-v1",
        "config_sha256": "b" * 64,
        "code_commit": "c" * 40,
        "threshold_source": "twin_validation_frozen",
        "training_data_origin": "twin",
        "input_fields": ["rgb"],
    }


def _predicted_locators(tmp_path: Path) -> dict[str, object]:
    legacy = _legacy_locators(tmp_path)
    return {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": _producer(),
        "images": legacy["images"],
    }


def test_valid_rgb_only_predictions_flow_with_bound_producer(tmp_path):
    from fluidmix.vision_measurement import evaluate_model_plan

    measured = evaluate_model_plan(_plan(), _predicted_locators(tmp_path))

    assert measured["mask_source"] == "rgb_only_prediction"
    assert measured["mask_prediction_producer"] == _producer()
    assert measured["denominators"]["valid_image_measurements"] == 9
    assert measured["decision_records"][0]["disposition"] == "keep"


def test_empty_predicted_mask_is_invalid_and_forces_abstention(tmp_path):
    from fluidmix.vision_measurement import evaluate_model_plan

    locators = _predicted_locators(tmp_path)
    locator = locators["images"]["TARGET-STAGE2"]
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8), mode="L").save(
        locator["mask_path"]
    )
    locator["mask_sha256"] = _sha(Path(locator["mask_path"]))

    measured = evaluate_model_plan(_plan(), locators)

    target = next(
        row for row in measured["image_measurements"] if row["image_id"] == "TARGET-STAGE2"
    )
    assert target["validity"] == "invalid"
    assert target["limit_codes"] == ["mask_empty"]
    assert measured["decision_records"][0]["disposition"] == "insufficient_evidence"
    assert measured["denominators"]["invalid_image_measurement_reasons"] == {
        "mask_empty": 1
    }


@pytest.mark.parametrize(
    "hidden_key",
    [
        "oracle",
        "label",
        "residual_volume",
        "residual_volume_ml",
        "palette",
        "composition",
    ],
)
def test_prediction_locator_rejects_hidden_fields_anywhere(tmp_path, hidden_key):
    from fluidmix.vision_measurement import evaluate_model_plan

    locators = _predicted_locators(tmp_path)
    locators["producer"][hidden_key] = "hidden"

    with pytest.raises(ValueError, match="forbidden prediction locator field"):
        evaluate_model_plan(_plan(), locators)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda producer: producer.update(model_sha256="A" * 64),
        lambda producer: producer.update(code_commit="c" * 39),
        lambda producer: producer.update(input_fields=["rgb", "semantic_aov"]),
        lambda producer: producer.update(producer_kind="semantic_segmenter"),
    ],
)
def test_prediction_producer_identity_hash_and_type_drift_is_rejected(
    tmp_path, mutation
):
    from fluidmix.vision_measurement import evaluate_model_plan

    locators = _predicted_locators(tmp_path)
    mutation(locators["producer"])

    with pytest.raises(ValueError, match="prediction producer"):
        evaluate_model_plan(_plan(), locators)


def test_legacy_schema_one_remains_compatible(tmp_path):
    from fluidmix.vision_measurement import evaluate_model_plan

    measured = evaluate_model_plan(_plan(), _legacy_locators(tmp_path))

    assert measured["decision_records"][0]["disposition"] == "keep"
    assert "mask_source" not in measured
    assert "mask_prediction_producer" not in measured


def test_writer_binds_prediction_manifest_and_producer(tmp_path):
    from fluidmix.vision_measurement import write_measured_evidence

    plan_path = tmp_path / "model-input-plan.json"
    locator_path = tmp_path / "predicted-mask-locators.json"
    output_path = tmp_path / "measured-evidence.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    locator_path.write_text(
        json.dumps(_predicted_locators(tmp_path / "pixels")), encoding="utf-8"
    )

    measured = write_measured_evidence(plan_path, locator_path, output_path)

    assert measured["mask_source"] == "rgb_only_prediction"
    assert measured["mask_prediction_producer"] == _producer()
    assert measured["source_model_input_sha256"] == _sha(plan_path)
    assert measured["source_image_locators_sha256"] == _sha(locator_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == measured

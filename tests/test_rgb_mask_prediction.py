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


def _temporal_predicted_locators(tmp_path: Path) -> dict[str, object]:
    artifact = _predicted_locators(tmp_path)
    images: dict[str, dict[str, str]] = {}
    for image_id, locator in artifact["images"].items():
        reference_path = (tmp_path / f"{image_id}-reference-rgb.png").resolve()
        Image.new("RGB", (4, 3), color=(17, 23, 31)).save(reference_path)
        images[image_id] = {
            "rgb_path": locator["rgb_path"],
            "rgb_sha256": locator["rgb_sha256"],
            "reference_rgb_path": str(reference_path),
            "reference_rgb_sha256": _sha(reference_path),
            "mask_path": locator["mask_path"],
            "mask_sha256": locator["mask_sha256"],
        }
    return {
        "schema_version": 3,
        "artifact_type": "rgb_temporal_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": _producer(),
        "images": images,
    }


def test_valid_rgb_only_predictions_flow_with_bound_producer(tmp_path):
    from fluidmix.vision_measurement import (
        evaluate_model_plan,
        trusted_measurement_bindings_from_result_v1,
    )

    measured = evaluate_model_plan(_plan(), _predicted_locators(tmp_path))

    assert measured["mask_source"] == "rgb_only_prediction"
    assert measured["mask_prediction_producer"] == _producer()
    assert len(measured["trusted_predicted_mask_registry_sha256"]) == 64
    assert all(
        row["trusted_predicted_mask_binding"]["image_id"] == row["image_id"]
        and row["trusted_predicted_mask_binding"]["measurement_id"]
        == f"MEAS-{row['image_id']}"
        and row["trusted_predicted_mask_binding"]["evidence_id"]
        == f"EVID-{row['image_id']}"
        and row["trusted_predicted_mask_binding"]["registry_sha256"]
        == measured["trusted_predicted_mask_registry_sha256"]
        for row in measured["image_measurements"]
    )
    assert measured["denominators"]["valid_image_measurements"] == 9
    assert measured["decision_records"][0]["disposition"] == "keep"
    live_bindings = trusted_measurement_bindings_from_result_v1(measured)
    assert len(live_bindings) == 9
    assert all(binding.validity == "valid" for binding in live_bindings)
    assert "TrustedPredictedMeasurementBindingV1" not in json.dumps(measured)


def test_public_measurement_adapter_flows_live_capabilities_to_part_b_projector(
    tmp_path,
):
    from fluidmix.part_b_evidence import (
        bind_part_b_measurements_from_adapter_v1,
        project_part_b_model_evidence_v1,
    )
    from fluidmix.vision_measurement import evaluate_model_plan

    measured = evaluate_model_plan(_plan(), _predicted_locators(tmp_path))
    templates = []
    for ordinal, tank in enumerate(("t200", "t400", "t100"), start=1):
        image_id = f"TARGET-STAGE{ordinal}"
        templates.append(
            {
                "run_id": "RUN-TARGET",
                "case_id": "CASE-TARGET-V1",
                "vial_ordinal": 1,
                "block_recipe_id": "S0117",
                "block_position": 1,
                "recipe_id": "S0117",
                "previous_recipe_id": "S0437",
                "recipe_ml": {"t100": 10.0, "t200": 40.0, "t400": 25.0},
                "dispense_order": ["t200", "t400", "t100"],
                "split_role": "train",
                "split_version": "whole-run-v1",
                "stage_id": f"STAGE{ordinal}",
                "tank_id": tank,
                "stage_ordinal": ordinal,
                "fill_clip_path": "clips/RUN-TARGET-V1.mp4",
                "fill_frame_range": [10, 20],
                "transition_frame_ranges": {"entry": [5, 10], "exit": [20, 25]},
                "clip_sha256": "1" * 64,
                "image_id": image_id,
                "rgb_sha256": "0" * 64,
                "mask_sha256": "0" * 64,
                "mask_source": "predicted",
                "measurement_id": "UNBOUND",
                "median_lab": [0.0, 0.0, 0.0],
                "mad_lab": [0.0, 0.0, 0.0],
                "iqr_lab": [0.0, 0.0, 0.0],
                "pairwise_delta_e76": [],
                "colour_regime_id": None,
                "evidence_id": "UNBOUND",
                "decision_id": None,
                "disposition": None,
                "limit_codes": [],
                "history_refs": [],
                "explanation_id": None,
                "model_revision": None,
                "prompt_sha256": None,
                "image_ids_sent": [],
                "guard_result": None,
            }
        )
    records = bind_part_b_measurements_from_adapter_v1(templates, measured)

    encoded = project_part_b_model_evidence_v1(
        records,
        phase="measured",
        predicted_mask_registry=measured,
        measurement_bindings=measured,
    )

    assert json.loads(encoded) == records
    assert all(record["measurement_id"].startswith("MEAS-TARGET-") for record in records)
    assert all(record["rgb_sha256"] != "0" * 64 for record in records)


def test_temporal_prediction_schema_binds_reference_rgb_without_changing_schema_two(
    tmp_path,
):
    from fluidmix.rgb_mask_prediction import (
        validate_rgb_mask_prediction_locators,
        validate_temporal_rgb_mask_prediction_locators,
    )

    spatial = _predicted_locators(tmp_path / "spatial")
    temporal = _temporal_predicted_locators(tmp_path / "temporal")

    spatial_images, spatial_producer = validate_rgb_mask_prediction_locators(spatial)
    temporal_images, temporal_producer = (
        validate_temporal_rgb_mask_prediction_locators(temporal)
    )

    assert spatial_images == spatial["images"]
    assert spatial_producer == spatial["producer"]
    assert temporal_images == temporal["images"]
    assert temporal_producer == temporal["producer"]
    assert all(
        set(locator)
        == {
            "rgb_path",
            "rgb_sha256",
            "reference_rgb_path",
            "reference_rgb_sha256",
            "mask_path",
            "mask_sha256",
        }
        for locator in temporal_images.values()
    )


def test_prediction_validator_treats_dynamic_image_ids_as_ids_not_field_names(tmp_path):
    from fluidmix.rgb_mask_prediction import validate_rgb_mask_prediction_locators

    artifact = _predicted_locators(tmp_path)
    locator = artifact["images"].pop("TARGET-STAGE1")
    artifact["images"]["label-free-camera-stage1"] = locator

    images, _ = validate_rgb_mask_prediction_locators(artifact)

    assert "label-free-camera-stage1" in images


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

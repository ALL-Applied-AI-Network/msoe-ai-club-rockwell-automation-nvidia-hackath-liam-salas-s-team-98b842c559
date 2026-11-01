"""Deterministic pixel-to-evidence tests for the three-vial vision plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_locator(
    root: Path,
    image_id: str,
    rgb_value: tuple[int, int, int],
    *,
    mask: np.ndarray | None = None,
    unique_bytes: bool = True,
) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    rgb_path = root / f"{image_id}-rgb.png"
    mask_path = root / f"{image_id}-mask.png"
    rgb = np.empty((3, 4, 3), dtype=np.uint8)
    rgb[:] = rgb_value
    if unique_bytes:
        rgb[2, 3] = tuple(hashlib.sha256(image_id.encode()).digest()[:3])
    if mask is None:
        mask = np.array(
            [[0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]],
            dtype=np.uint8,
        )
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    return {
        "rgb_path": str(rgb_path),
        "rgb_sha256": _sha256(rgb_path),
        "mask_path": str(mask_path),
        "mask_sha256": _sha256(mask_path),
    }


def _stage(stage_index: int, target_id: str, peer_ids: list[str]) -> dict:
    stage_id = f"STAGE{stage_index}"
    return {
        "evidence_id": f"EVID-{target_id}",
        "measurement_id": f"MEAS-{target_id}",
        "stage_id": stage_id,
        "stage_ordinal": stage_index,
        "time_code": (171, 342, 686)[stage_index - 1],
        "target_image_id": target_id,
        "reference_cases": [
            {
                "case_id": "all_reference_batches",
                "mode": "all_reference_batches",
                "alternative_id": None,
                "candidate_peers": [
                    {
                        "peer_image_id": peer_id,
                        "batch_id": f"REF-{peer_number}",
                        "origin": "reference_bank",
                    }
                    for peer_number, peer_id in enumerate(peer_ids, start=1)
                ],
                "disposition_source": True,
            }
        ],
        "primary_mode": "all_reference_batches",
        "unit": "delta_e76",
    }


def _model_plan(*, peer_count: int = 2) -> dict:
    stages = []
    for stage_index in range(1, 4):
        target = f"TARGET-STAGE{stage_index}"
        peers = [f"PEER{index}-STAGE{stage_index}" for index in range(1, peer_count + 1)]
        stages.append(_stage(stage_index, target, peers))
    return {
        "schema_version": 3,
        "artifact_type": "model_input_plan",
        "config_id": "three-vial-separated-reference-bank-v1",
        "status": {
            "config_status": "design_frozen",
            "render_status": "unrendered",
            "training_eligible": False,
        },
        "decision_records": [
            {
                "decision_id": "DEC-BATCH-V1",
                "family_id": "TR",
                "batch_id": "BATCH_A1B2C3D4E5F6",
                "cohort": "scored_target",
                "partition": "train",
                "previous_sim_recipe_id": "S0437",
                "current_sim_recipe_id": "S0117",
                "previous_programmed_ml": {"t100": 60, "t200": 10, "t400": 10},
                "current_programmed_ml": {"t100": 10, "t200": 40, "t400": 25},
                "programmed_dispense_order": ["t200", "t400", "t100"],
                "vial_id": "V1",
                "vial_position": 1,
                "stages": stages,
                "decision_timing": "retrospective_after_complete_three_vial_batch_no_latency_claim",
            }
        ],
        "real_data_used": False,
    }


def _f2_model_plan(*, peer_count: int = 2) -> dict:
    plan = _model_plan(peer_count=peer_count)
    plan["schema_version"] = 2
    plan.pop("status")
    config = {
        "tight_cluster_max_diameter_delta_e76": 2.0,
        "threshold_floor_delta_e76": 2.0,
        "threshold_margin_delta_e76": 1.0,
    }
    plan["decision_config"] = config
    plan["decision_config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decision = plan["decision_records"][0]
    decision["run_id"] = "RUN-TARGET"
    for stage in decision["stages"]:
        for peer_number, peer in enumerate(
            stage["reference_cases"][0]["candidate_peers"], start=1
        ):
            peer["run_id"] = f"RUN-REF-{peer_number}"
    return plan


def _locators(
    tmp_path: Path,
    *,
    target_rgb=(0, 0, 0),
    peer_count: int = 2,
    unique_bytes: bool = True,
) -> dict:
    images: dict[str, dict[str, str]] = {}
    for stage_index in range(1, 4):
        target_id = f"TARGET-STAGE{stage_index}"
        images[target_id] = _image_locator(
            tmp_path, target_id, target_rgb, unique_bytes=unique_bytes
        )
        for peer_index in range(1, peer_count + 1):
            peer_id = f"PEER{peer_index}-STAGE{stage_index}"
            images[peer_id] = _image_locator(
                tmp_path, peer_id, (0, 0, 0), unique_bytes=unique_bytes
            )
    return {"schema_version": 1, "data_origin": "twin", "images": images}


def _walk(value):
    yield value
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_measure_image_uses_masked_coordinate_median_lab(tmp_path):
    from fluidmix.measurement.vision_measurement import measure_image

    locator = _image_locator(tmp_path, "RED", (255, 0, 0))

    measured = measure_image("RED", locator)

    assert measured["validity"] == "valid"
    assert measured["liquid_pixel_count"] == 4
    assert measured["lab_raw"] == pytest.approx(
        [53.240588, 80.092308, 67.202751], abs=1e-5
    )
    assert measured["lab_6dp"] == ["53.240588", "80.092308", "67.202751"]
    assert measured["unit"] == "cie_lab_d65_2deg"


def test_pipeline_injects_reference_selection_into_measurement(tmp_path, monkeypatch):
    import fluidmix.pipeline.model_plan_measurement as pipeline_measurement

    calls = []
    original = pipeline_measurement.select_peer_reference

    def tracked(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_measurement, "select_peer_reference", tracked)

    result = pipeline_measurement.evaluate_model_plan(
        _model_plan(), _locators(tmp_path)
    )

    assert result["decision_records"][0]["disposition"] == "keep"
    assert len(calls) == 3


@pytest.mark.parametrize("target_rgb, expected", [((0, 0, 0), "keep"), ((255, 255, 255), "discard")])
def test_primary_cross_batch_stage_evidence_drives_keep_or_discard(
    tmp_path, target_rgb, expected
):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan

    result = evaluate_model_plan(_model_plan(), _locators(tmp_path, target_rgb=target_rgb))

    decision = result["decision_records"][0]
    assert decision["disposition"] == expected
    assert decision["disposition_source"] == "deterministic_pipeline_primary_reference_evidence"
    assert len(decision["stages"]) == 3
    assert all(stage["validity"] == "valid" for stage in decision["stages"])
    assert all(stage["unit"] == "delta_e76" for stage in decision["stages"])
    assert all(
        stage["stage_outlier"] is (expected == "discard")
        for stage in decision["stages"]
    )
    assert all(
        stage["target_delta_e76_6dp"]
        == f"{stage['target_delta_e76_raw']:.6f}"
        for stage in decision["stages"]
    )


def test_f2_repeated_content_on_distinct_paths_is_diagnosed_and_abstains(tmp_path):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan

    result = evaluate_model_plan(
        _f2_model_plan(),
        _locators(
            tmp_path, target_rgb=(255, 255, 255), unique_bytes=False
        ),
    )

    primary = result["decision_records"][0]["stages"][0]["reference_cases"][0]
    assert primary["validity"] == "insufficient_evidence"
    assert primary["limit_codes"] == ["duplicate_reference_rgb_bytes"]
    assert result["decision_records"][0]["disposition"] == "insufficient_evidence"
    assert result["artifact_content_diagnostics"] == {
        "rgb_file_sha256_total_count": 9,
        "rgb_file_sha256_unique_count": 2,
        "mask_file_sha256_total_count": 9,
        "mask_file_sha256_unique_count": 1,
    }


def test_f2_target_rgb_reused_as_reference_abstains(tmp_path):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan

    locators = _locators(tmp_path, unique_bytes=False)
    locators["images"]["PEER2-STAGE1"] = _image_locator(
        tmp_path, "PEER2-STAGE1-UNIQUE", (10, 10, 10)
    )

    result = evaluate_model_plan(_f2_model_plan(), locators)

    primary = result["decision_records"][0]["stages"][0]["reference_cases"][0]
    assert primary["validity"] == "insufficient_evidence"
    assert primary["limit_codes"] == ["target_rgb_bytes_reused_as_reference"]


@pytest.mark.parametrize("role", ["rgb", "mask"])
def test_f2_rejects_artifact_path_reuse(tmp_path, role):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan

    locators = _locators(tmp_path, target_rgb=(255, 255, 255))
    target = locators["images"]["TARGET-STAGE1"]
    peer = locators["images"]["PEER1-STAGE1"]
    peer[f"{role}_path"] = target[f"{role}_path"]

    with pytest.raises(ValueError, match=f"F2 {role} image paths must be unique"):
        evaluate_model_plan(_f2_model_plan(), locators)


def test_dynamic_masks_are_independent_while_thin_or_tied_reference_abstains(tmp_path):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan

    mismatched = _locators(tmp_path / "mismatch")
    different_mask = np.array(
        [[255, 0, 0, 0], [255, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
    )
    mismatched["images"]["PEER1-STAGE1"] = _image_locator(
        tmp_path / "mismatch", "PEER1-STAGE1-ALT", (0, 0, 0), mask=different_mask
    )
    mismatch_result = evaluate_model_plan(_model_plan(), mismatched)
    assert mismatch_result["decision_records"][0]["disposition"] == "keep"
    assert mismatch_result["decision_records"][0]["stages"][0]["limit_codes"] == []

    thin_result = evaluate_model_plan(
        _model_plan(peer_count=1), _locators(tmp_path / "thin", peer_count=1)
    )
    assert thin_result["decision_records"][0]["disposition"] == "insufficient_evidence"
    assert "reference_not_unique_or_too_small" in thin_result["decision_records"][0]["stages"][0]["limit_codes"]

    tied_plan = _model_plan(peer_count=4)
    tied_locators = _locators(tmp_path / "tied", peer_count=4)
    for stage_index in range(1, 4):
        for peer_index in (3, 4):
            peer_id = f"PEER{peer_index}-STAGE{stage_index}"
            tied_locators["images"][peer_id] = _image_locator(
                tmp_path / "tied", peer_id + "-WHITE", (255, 255, 255)
            )
    tied_result = evaluate_model_plan(tied_plan, tied_locators)
    assert tied_result["decision_records"][0]["disposition"] == "insufficient_evidence"
    assert "reference_not_unique_or_too_small" in tied_result["decision_records"][0]["stages"][0]["limit_codes"]


def test_missing_or_tampered_pixels_fail_closed(tmp_path):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan
    from fluidmix.measurement.vision_measurement import measure_image

    locator = _image_locator(tmp_path, "TAMPER", (0, 0, 0))
    Path(locator["rgb_path"]).write_bytes(b"not the frozen PNG")
    measured = measure_image("TAMPER", locator)
    assert measured["validity"] == "invalid"
    assert measured["limit_codes"] == ["rgb_hash_mismatch"]

    locators = _locators(tmp_path / "missing")
    Path(locators["images"]["TARGET-STAGE2"]["mask_path"]).unlink()
    result = evaluate_model_plan(_model_plan(), locators)
    decision = result["decision_records"][0]
    assert decision["disposition"] == "insufficient_evidence"
    assert "target_measurement_invalid" in decision["stages"][1]["limit_codes"]


def test_oracle_keys_and_values_are_rejected_and_never_emitted(tmp_path):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan

    plan = _model_plan()
    plan["decision_records"][0]["synthetic_profile_id"] = "first_only"
    with pytest.raises(ValueError, match="oracle"):
        evaluate_model_plan(plan, _locators(tmp_path / "key"))

    plan = _model_plan()
    plan["decision_records"][0]["family_id"] = "after_t200"
    with pytest.raises(ValueError, match="oracle"):
        evaluate_model_plan(plan, _locators(tmp_path / "value"))

    result = evaluate_model_plan(_model_plan(), _locators(tmp_path / "clean"))
    assert result["data_origin"] == "twin"
    walked = {item for item in _walk(result) if isinstance(item, str)}
    assert {
        "synthetic_profile_id",
        "appearance_blend_alpha",
        "synthetic_onset_id",
        "clean_stage_linear_rgb",
        "observed_stage_linear_rgb",
        "semantic_aov",
        "first_only",
        "after_t200",
    }.isdisjoint(walked)

    real_locators = _locators(tmp_path / "real")
    real_locators["data_origin"] = "real"
    with pytest.raises(ValueError, match="twin"):
        evaluate_model_plan(_model_plan(), real_locators)


@pytest.mark.parametrize(
    "oracle_key, oracle_value",
    [
        ("residual_particle_count", 56),
        ("residual_particle_plan", {"residual_particle_count": 56}),
        ("residual_source_component_counts", {"t200": 7, "t400": 7, "t100": 42}),
        ("expected_disposition", "discard"),
        ("label_source", "prescribed_synthetic_latent_oracle"),
    ],
)
def test_residual_scoring_oracles_are_rejected_from_model_input(
    tmp_path, oracle_key, oracle_value
):
    from fluidmix.pipeline.model_plan_measurement import evaluate_model_plan

    plan = _model_plan()
    plan["decision_records"][0][oracle_key] = oracle_value

    with pytest.raises(ValueError, match="oracle"):
        evaluate_model_plan(plan, _locators(tmp_path / oracle_key))


def test_writer_publishes_new_evidence_without_clobber(tmp_path):
    from fluidmix.pipeline.model_plan_measurement import write_measured_evidence

    plan_path = tmp_path / "model-input-plan.json"
    locator_path = tmp_path / "image-locators.json"
    output_path = tmp_path / "measured-evidence.json"
    plan_path.write_text(json.dumps(_model_plan()), encoding="utf-8")
    locator_path.write_text(json.dumps(_locators(tmp_path / "pixels")), encoding="utf-8")

    record = write_measured_evidence(plan_path, locator_path, output_path)

    assert output_path.is_file()
    assert record == json.loads(output_path.read_text(encoding="utf-8"))
    assert record["source_model_input_sha256"] == _sha256(plan_path)
    assert record["source_image_locators_sha256"] == _sha256(locator_path)
    with pytest.raises(FileExistsError):
        write_measured_evidence(plan_path, locator_path, output_path)

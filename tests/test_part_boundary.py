"""Production Part B evidence and provenance boundary tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


SIM_METADATA_FIELDS = {
    "tank_colour_preset",
    "resolved_tank_transmission_1cm",
    "advanced_toggles",
    "tank_colour_preset_hash",
    "renderer_palette",
    "renderer_seed",
    "residue_ml",
    "residue_source_liquid",
    "particle_state_summary",
    "label",
    "host_profile",
    "excluded_from_model_inputs",
}


def test_deploy_policy_accepts_exact_recipe_tanks_and_rejects_sim_metadata_fields():
    from fluidmix.deploy_input_policy import validate_deploy_input_fields

    positive = {
        "recipe_ml": {"t100": 20.0, "t200": 10.0, "t400": 30.0},
    }
    validate_deploy_input_fields(positive)

    for field in sorted(SIM_METADATA_FIELDS):
        contaminated = deepcopy(positive)
        contaminated[field] = True
        with pytest.raises(ValueError, match="forbidden_deploy_input_field"):
            validate_deploy_input_fields(contaminated)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_artifact(
    tmp_path: Path,
    *,
    image_ids: tuple[str, ...] = ("IMAGE-1",),
    empty_mask: bool = False,
) -> dict[str, object]:
    images = {}
    for index, image_id in enumerate(image_ids, start=1):
        rgb = tmp_path / f"{image_id}-source-rgb.png"
        mask = tmp_path / f"{image_id}-predicted-mask.png"
        Image.new("RGB", (4, 3), color=(17 + index, 23, 31)).save(rgb)
        pixels = np.zeros((3, 4), dtype=np.uint8)
        if not empty_mask:
            pixels[1:, 1:3] = 255
        Image.fromarray(pixels, mode="L").save(mask)
        images[image_id] = {
            "rgb_path": str(rgb.resolve()),
            "rgb_sha256": _sha(rgb),
            "mask_path": str(mask.resolve()),
            "mask_sha256": _sha(mask),
        }
    return {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": {
            "producer_id": "segmenter-1",
            "producer_kind": "rgb_only_liquid_segmenter",
            "model_id": "model-1",
            "model_sha256": "a" * 64,
            "config_id": "config-1",
            "config_sha256": "b" * 64,
            "code_commit": "c" * 40,
            "threshold_source": "twin_validation_frozen",
            "training_data_origin": "twin",
            "input_fields": ["rgb"],
        },
        "images": images,
    }


def test_trusted_prediction_registry_is_factory_only_and_measurement_retains_binding(
    tmp_path,
):
    import fluidmix.rgb_mask_prediction as prediction
    from fluidmix.vision_measurement import measure_trusted_predicted_image

    registry = prediction.build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(tmp_path)
    )
    entry = registry.entry("IMAGE-1")
    measured, binding = measure_trusted_predicted_image(
        entry,
        measurement_id="MEASUREMENT-1",
        evidence_id="EVIDENCE-1",
    )

    assert measured["validity"] == "valid"
    assert binding.image_id == "IMAGE-1"
    assert binding.measurement_id == "MEASUREMENT-1"
    assert binding.evidence_id == "EVIDENCE-1"
    assert binding.registry_sha256 == registry.registry_sha256
    assert entry.model_sha256 == "a" * 64
    assert entry.config_sha256 == "b" * 64
    assert entry.code_commit == "c" * 40
    assert entry.threshold_source == "twin_validation_frozen"

    raw_locator = _prediction_artifact(tmp_path)["images"]["IMAGE-1"]
    with pytest.raises(TypeError, match="trusted predicted-mask entry"):
        measure_trusted_predicted_image(
            raw_locator,
            measurement_id="MEASUREMENT-1",
            evidence_id="EVIDENCE-1",
        )


def _measured_sequence(registry) -> tuple[list[dict[str, object]], list[object]]:
    from fluidmix.vision_measurement import measure_trusted_predicted_image

    records = []
    bindings = []
    order = ("t200", "t400", "t100")
    for ordinal, tank in enumerate(order, start=1):
        image_id = f"IMAGE-{ordinal}"
        measurement_id = f"MEASUREMENT-{ordinal}"
        evidence_id = f"EVIDENCE-{ordinal}"
        _, binding = measure_trusted_predicted_image(
            registry.entry(image_id),
            measurement_id=measurement_id,
            evidence_id=evidence_id,
        )
        bindings.append(binding)
        records.append(
            {
                "run_id": "RUN-1",
                "case_id": "CASE-1",
                "vial_ordinal": 1,
                "block_recipe_id": "RECIPE-1",
                "block_position": 1,
                "recipe_id": "RECIPE-1",
                "previous_recipe_id": "RECIPE-0",
                "recipe_ml": {"t100": 20.0, "t200": 10.0, "t400": 30.0},
                "dispense_order": list(order),
                "split_role": "train",
                "split_version": "whole-run-v1",
                "stage_id": f"STAGE-{ordinal}",
                "tank_id": tank,
                "stage_ordinal": ordinal,
                "fill_clip_path": "clips/RUN-1-CASE-1.mp4",
                "fill_frame_range": [10, 20],
                "transition_frame_ranges": {"entry": [5, 10], "exit": [20, 25]},
                "clip_sha256": "1" * 64,
                "image_id": image_id,
                "rgb_sha256": registry.entry(image_id).rgb_sha256,
                "mask_sha256": registry.entry(image_id).mask_sha256,
                "mask_source": "predicted",
                "measurement_id": measurement_id,
                "median_lab": [50.0, 1.0, -2.0],
                "mad_lab": [0.5, 0.25, 0.75],
                "iqr_lab": [1.0, 0.5, 1.5],
                "pairwise_delta_e76": [],
                "colour_regime_id": None,
                "evidence_id": evidence_id,
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
    return records, bindings


def _trusted_sequence(tmp_path: Path):
    from fluidmix.rgb_mask_prediction import build_trusted_predicted_mask_registry_v1

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(
            tmp_path,
            image_ids=("IMAGE-1", "IMAGE-2", "IMAGE-3"),
        )
    )
    records, bindings = _measured_sequence(registry)
    return registry, records, bindings


def test_projector_emits_only_ordered_flat_production_records(tmp_path):
    from fluidmix.part_b_evidence import project_part_b_model_evidence_v1

    registry, records, bindings = _trusted_sequence(tmp_path)
    encoded = project_part_b_model_evidence_v1(
        records,
        phase="measured",
        predicted_mask_registry=registry,
        measurement_bindings=bindings,
    )

    assert json.loads(encoded) == records
    assert encoded.startswith('[{"block_position":1')
    assert "registry_sha256" not in encoded
    assert "entry_sha256" not in encoded
    assert [row["stage_ordinal"] for row in json.loads(encoded)] == [1, 2, 3]


def test_trusted_registry_rejects_a_predicted_label_over_an_oracle_mask_path(tmp_path):
    from fluidmix.rgb_mask_prediction import build_trusted_predicted_mask_registry_v1

    artifact = _prediction_artifact(tmp_path)
    locator = artifact["images"]["IMAGE-1"]
    oracle = tmp_path / "masks" / "oracle" / "IMAGE-1.png"
    oracle.parent.mkdir(parents=True)
    oracle.write_bytes(Path(locator["mask_path"]).read_bytes())
    locator["mask_path"] = str(oracle.resolve())
    locator["mask_sha256"] = _sha(oracle)

    with pytest.raises(ValueError, match="oracle mask path"):
        build_trusted_predicted_mask_registry_v1(artifact)


@pytest.mark.parametrize(
    "mutation",
    ["oracle", "missing_binding", "hash_mismatch", "caller_registry"],
)
def test_projector_returns_insufficient_evidence_for_untrusted_provenance(
    tmp_path, mutation
):
    from fluidmix.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )

    registry, records, bindings = _trusted_sequence(tmp_path)
    registry_input = registry
    if mutation == "oracle":
        records[0]["mask_source"] = "oracle"
    elif mutation == "missing_binding":
        bindings.pop()
    elif mutation == "hash_mismatch":
        records[0]["mask_sha256"] = "f" * 64
    else:
        registry_input = {
            record["image_id"]: registry.entry(record["image_id"]).locator()
            for record in records
        }

    with pytest.raises(PartBInsufficientEvidence) as caught:
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry_input,
            measurement_bindings=bindings,
        )
    assert caught.value.disposition == "insufficient_evidence"
    assert "fallback" not in str(caught.value)


@pytest.mark.parametrize("failure", ["missing", "unusable"])
def test_projector_abstains_when_predicted_mask_pixels_are_unavailable(
    tmp_path, failure
):
    from fluidmix.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )
    from fluidmix.rgb_mask_prediction import build_trusted_predicted_mask_registry_v1

    artifact = _prediction_artifact(
        tmp_path,
        image_ids=("IMAGE-1", "IMAGE-2", "IMAGE-3"),
        empty_mask=failure == "unusable",
    )
    registry = build_trusted_predicted_mask_registry_v1(artifact)
    if failure == "missing":
        Path(artifact["images"]["IMAGE-2"]["mask_path"]).unlink()
    records, bindings = _measured_sequence(registry)

    with pytest.raises(
        PartBInsufficientEvidence,
        match="predicted_measurement_unusable",
    ):
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry,
            measurement_bindings=bindings,
        )


def test_projector_rejects_every_sim_metadata_property_before_serialization(tmp_path):
    from fluidmix.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )

    registry, records, bindings = _trusted_sequence(tmp_path)
    for field in sorted(SIM_METADATA_FIELDS):
        contaminated = deepcopy(records)
        contaminated[0][field] = True
        with pytest.raises(PartBInsufficientEvidence):
            project_part_b_model_evidence_v1(
                contaminated,
                phase="measured",
                predicted_mask_registry=registry,
                measurement_bindings=bindings,
            )


@pytest.mark.parametrize(
    "value",
    [pytest.param("missing", id="omitted"), "base", "a" * 64, "arbitrary"],
)
def test_projector_requires_explicit_null_colour_regime(tmp_path, value):
    from fluidmix.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )

    registry, records, bindings = _trusted_sequence(tmp_path)
    for record in records:
        if value == "missing":
            record.pop("colour_regime_id")
        else:
            record["colour_regime_id"] = value
    with pytest.raises(PartBInsufficientEvidence):
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry,
            measurement_bindings=bindings,
        )


def test_model_plan_context_uses_exact_envelopes_not_recipe_leaf_names():
    from fluidmix.model_input_plan_builder import ModelInputPlanError, _validate_context

    context = {
        "schema_version": 1,
        "artifact_type": "stage_batch_context",
        "split_version": "inn-7-splits-v2",
        "batches": [
            {
                "batch_id": "BATCH-1",
                "cohort": "scored_target",
                "family_id": "VA",
                "current_recipe_id": "R-1",
                "previous_recipe_id": "R-0",
                "run_id": "RUN-1",
                "programmed_recipe": {
                    "dispense_order": ["t200", "t400", "t100"],
                    "previous_ml_by_tank": {
                        "t100": 20,
                        "t200": 10,
                        "t400": 30,
                    },
                    "current_ml_by_tank": {
                        "t100": 20,
                        "t200": 10,
                        "t400": 30,
                    },
                },
            }
        ],
    }
    assert _validate_context(context)[0]["programmed_recipe"][
        "current_ml_by_tank"
    ] == {"t100": 20.0, "t200": 10.0, "t400": 30.0}

    for field in sorted(SIM_METADATA_FIELDS):
        contaminated = deepcopy(context)
        contaminated[field] = True
        with pytest.raises(ModelInputPlanError, match="forbidden_part_a_envelope_field"):
            _validate_context(contaminated)

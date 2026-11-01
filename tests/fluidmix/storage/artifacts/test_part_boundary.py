"""Production Part B evidence and provenance boundary tests."""

from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[4]
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
    from fluidmix.core.policy import validate_deploy_input_fields

    positive = {
        "recipe_ml": {"t100": 20.0, "t200": 10.0, "t400": 30.0},
    }
    validate_deploy_input_fields(positive)

    for field in sorted(SIM_METADATA_FIELDS):
        contaminated = deepcopy(positive)
        contaminated[field] = True
        with pytest.raises(ValueError, match="forbidden_deploy_input_field"):
            validate_deploy_input_fields(contaminated)


def test_deploy_policy_validates_exact_part_b_envelope_paths(tmp_path):
    from fluidmix.core.policy import validate_deploy_input_fields

    _, records, _ = _trusted_sequence(tmp_path)
    validate_deploy_input_fields(records[0], envelope="part_b_evidence_v1")

    nested = deepcopy(records[0])
    nested["recipe_ml"]["label"] = 1.0
    with pytest.raises(ValueError, match="forbidden_deploy_input_field: recipe_ml.label"):
        validate_deploy_input_fields(nested, envelope="part_b_evidence_v1")

    contaminated = deepcopy(records[0])
    contaminated["renderer_palette"] = {}
    with pytest.raises(
        ValueError,
        match="forbidden_deploy_input_field: renderer_palette",
    ):
        validate_deploy_input_fields(
            contaminated,
            envelope="part_b_evidence_v1",
        )


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
    import fluidmix.perception.masks as prediction
    from fluidmix.measurement.vision_measurement import measure_trusted_predicted_image

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


@pytest.mark.parametrize(
    ("capability", "field", "replacement"),
    [
        ("entry", "_mask_path", "/tmp/oracle-mask.png"),
        ("entry", "_mask_sha256", "f" * 64),
        ("registry", "_registry_sha256", "e" * 64),
    ],
)
def test_trusted_prediction_capabilities_reject_attribute_mutation(
    tmp_path, capability, field, replacement
):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(tmp_path)
    )
    target = registry.entry("IMAGE-1") if capability == "entry" else registry

    with pytest.raises(AttributeError, match="immutable"):
        setattr(target, field, replacement)


@pytest.mark.parametrize(
    ("capability", "field"),
    [
        ("entry", "_mask_path"),
        ("registry", "_registry_sha256"),
        ("measurement", "_measurement_sha256"),
    ],
)
def test_trusted_capabilities_reject_attribute_deletion(tmp_path, capability, field):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1
    from fluidmix.measurement.vision_measurement import measure_trusted_predicted_image

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(tmp_path)
    )
    entry = registry.entry("IMAGE-1")
    _, binding = measure_trusted_predicted_image(
        entry,
        measurement_id="MEASUREMENT-1",
        evidence_id="EVIDENCE-1",
    )
    targets = {"entry": entry, "registry": registry, "measurement": binding}

    with pytest.raises(AttributeError, match="immutable"):
        delattr(targets[capability], field)


def test_measurement_adapter_classifies_missing_entry_attribute_as_untrusted(tmp_path):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1
    from fluidmix.measurement.vision_measurement import measure_trusted_predicted_image

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(tmp_path)
    )
    entry = registry.entry("IMAGE-1")
    object.__delattr__(entry, "_mask_path")

    with pytest.raises(TypeError, match="trusted predicted-mask entry"):
        measure_trusted_predicted_image(
            entry,
            measurement_id="MEASUREMENT-1",
            evidence_id="EVIDENCE-1",
        )


@pytest.mark.parametrize("registry_link", [None, "A" * 64, "a" * 63])
def test_measurement_rejects_invalid_registry_link_before_pixel_reads(
    tmp_path, monkeypatch, registry_link
):
    from fluidmix.perception.masks import (
        build_trusted_predicted_mask_registry_v1,
        is_trusted_predicted_mask_entry_v1,
    )
    from fluidmix.measurement.vision_measurement import measure_trusted_predicted_image

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(tmp_path)
    )
    entry = registry.entry("IMAGE-1")
    if registry_link is None:
        object.__delattr__(entry, "_registry_sha256")
    else:
        object.__setattr__(entry, "_registry_sha256", registry_link)
    read_count = 0
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path):
        nonlocal read_count
        read_count += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    assert is_trusted_predicted_mask_entry_v1(entry) is False
    with pytest.raises(TypeError, match="trusted predicted-mask entry"):
        measure_trusted_predicted_image(
            entry,
            measurement_id="MEASUREMENT-1",
            evidence_id="EVIDENCE-1",
        )
    assert read_count == 0


def test_projector_classifies_missing_registry_attribute_as_insufficient(tmp_path):
    from fluidmix.storage.artifacts.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )

    registry, records, bindings = _trusted_sequence(tmp_path)
    object.__delattr__(registry, "_registry_sha256")

    with pytest.raises(
        PartBInsufficientEvidence,
        match="trusted_predicted_mask_registry_missing",
    ):
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry,
            measurement_bindings=bindings,
        )


def test_projector_classifies_missing_measurement_attribute_as_insufficient(tmp_path):
    from fluidmix.storage.artifacts.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )

    registry, records, bindings = _trusted_sequence(tmp_path)
    object.__delattr__(bindings[0], "_measurement_sha256")

    with pytest.raises(
        PartBInsufficientEvidence,
        match="untrusted_measurement_binding",
    ):
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry,
            measurement_bindings=bindings,
        )


def test_trusted_prediction_registry_nested_producer_state_is_immutable(tmp_path):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(tmp_path)
    )

    with pytest.raises(AttributeError):
        registry._producer["input_fields"].append("oracle")
    assert registry.producer["input_fields"] == ["rgb"]


def test_measurement_revalidates_entry_digest_after_forced_locator_swap(tmp_path):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1
    from fluidmix.measurement.vision_measurement import measure_trusted_predicted_image

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(tmp_path)
    )
    entry = registry.entry("IMAGE-1")
    oracle_mask = tmp_path / "oracle-mask.png"
    Image.fromarray(np.full((3, 4), 255, dtype=np.uint8), mode="L").save(
        oracle_mask
    )
    object.__setattr__(entry, "_mask_path", str(oracle_mask.resolve()))
    object.__setattr__(entry, "_mask_sha256", _sha(oracle_mask))

    with pytest.raises(TypeError, match="trusted predicted-mask entry"):
        measure_trusted_predicted_image(
            entry,
            measurement_id="MEASUREMENT-1",
            evidence_id="EVIDENCE-1",
        )


def test_projector_revalidates_registry_digest_after_forced_entry_mutation(tmp_path):
    from fluidmix.storage.artifacts.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )

    registry, records, bindings = _trusted_sequence(tmp_path)
    object.__setattr__(
        registry.entry("IMAGE-1"),
        "_mask_path",
        str((tmp_path / "oracle-mask.png").resolve()),
    )

    with pytest.raises(
        PartBInsufficientEvidence,
        match="trusted_predicted_mask_registry_missing",
    ):
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry,
            measurement_bindings=bindings,
        )


def test_trusted_measurement_uses_each_hash_verified_file_read_exactly_once(
    tmp_path, monkeypatch
):
    import fluidmix.measurement.vision_measurement as measurement_module
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

    artifact = _prediction_artifact(tmp_path)
    locator = artifact["images"]["IMAGE-1"]
    rgb_path = Path(locator["rgb_path"])
    mask_path = Path(locator["mask_path"])
    rgb = np.full((3, 4, 3), (10, 10, 200), dtype=np.uint8)
    rgb[1:, 1:3] = (200, 10, 10)
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    locator["rgb_sha256"] = _sha(rgb_path)
    registry = build_trusted_predicted_mask_registry_v1(artifact)

    replacement_mask = np.zeros((3, 4), dtype=np.uint8)
    replacement_mask[0, :] = 255
    original_read_bytes = Path.read_bytes
    read_counts = {rgb_path: 0, mask_path: 0}

    def read_then_replace_mask(path):
        data = original_read_bytes(path)
        if path in read_counts:
            read_counts[path] += 1
        if path == mask_path and read_counts[path] == 1:
            Image.fromarray(replacement_mask, mode="L").save(mask_path)
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_replace_mask)

    measured, _ = measurement_module.measure_trusted_predicted_image(
        registry.entry("IMAGE-1"),
        measurement_id="MEASUREMENT-1",
        evidence_id="EVIDENCE-1",
    )

    assert read_counts == {rgb_path: 1, mask_path: 1}
    assert measured["mad_lab"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def _measured_sequence(registry) -> tuple[list[dict[str, object]], list[object]]:
    from fluidmix.measurement.vision_measurement import measure_trusted_predicted_image

    records = []
    bindings = []
    order = ("t200", "t400", "t100")
    for ordinal, tank in enumerate(order, start=1):
        image_id = f"IMAGE-{ordinal}"
        measurement_id = f"MEASUREMENT-{ordinal}"
        evidence_id = f"EVIDENCE-{ordinal}"
        measurement, binding = measure_trusted_predicted_image(
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
                "median_lab": measurement.get("median_lab", [0.0, 0.0, 0.0]),
                "mad_lab": measurement.get("mad_lab", [0.0, 0.0, 0.0]),
                "iqr_lab": measurement.get("iqr_lab", [0.0, 0.0, 0.0]),
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
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(
            tmp_path,
            image_ids=("IMAGE-1", "IMAGE-2", "IMAGE-3"),
        )
    )
    records, bindings = _measured_sequence(registry)
    return registry, records, bindings


def test_projector_emits_only_ordered_flat_production_records(tmp_path):
    from fluidmix.storage.artifacts.part_b_evidence import project_part_b_model_evidence_v1

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


@pytest.mark.parametrize("field", ["median_lab", "mad_lab", "iqr_lab"])
def test_projector_rejects_mutated_quantitative_measurement_content(
    tmp_path, field
):
    from fluidmix.storage.artifacts.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )

    registry, records, bindings = _trusted_sequence(tmp_path)
    records[0][field][0] += 1.0

    with pytest.raises(
        PartBInsufficientEvidence,
        match="measurement_content_mismatch",
    ):
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry,
            measurement_bindings=bindings,
        )


def test_invalid_measurement_capability_cannot_be_mutated_valid(tmp_path):
    from fluidmix.storage.artifacts.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

    registry = build_trusted_predicted_mask_registry_v1(
        _prediction_artifact(
            tmp_path,
            image_ids=("IMAGE-1", "IMAGE-2", "IMAGE-3"),
            empty_mask=True,
        )
    )
    records, bindings = _measured_sequence(registry)
    assert bindings[0].validity == "invalid"

    with pytest.raises((AttributeError, TypeError)):
        bindings[0]._validity = "valid"
    with pytest.raises(PartBInsufficientEvidence, match="predicted_measurement_unusable"):
        project_part_b_model_evidence_v1(
            records,
            phase="measured",
            predicted_mask_registry=registry,
            measurement_bindings=bindings,
        )


def test_trusted_registry_rejects_a_predicted_label_over_an_oracle_mask_path(tmp_path):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

    artifact = _prediction_artifact(tmp_path)
    locator = artifact["images"]["IMAGE-1"]
    oracle = tmp_path / "masks" / "oracle" / "IMAGE-1.png"
    oracle.parent.mkdir(parents=True)
    oracle.write_bytes(Path(locator["mask_path"]).read_bytes())
    locator["mask_path"] = str(oracle.resolve())
    locator["mask_sha256"] = _sha(oracle)

    with pytest.raises(ValueError, match="oracle mask path"):
        build_trusted_predicted_mask_registry_v1(artifact)


def test_trusted_registry_rejects_mask_symlink_resolving_to_oracle_path(tmp_path):
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

    artifact = _prediction_artifact(tmp_path)
    locator = artifact["images"]["IMAGE-1"]
    oracle = tmp_path / "masks" / "oracle" / "IMAGE-1.png"
    oracle.parent.mkdir(parents=True)
    oracle.write_bytes(Path(locator["mask_path"]).read_bytes())
    innocent_link = tmp_path / "IMAGE-1-predicted-mask-link.png"
    innocent_link.symlink_to(oracle)
    locator["mask_path"] = str(innocent_link)
    locator["mask_sha256"] = _sha(innocent_link)

    with pytest.raises(ValueError, match="oracle mask path"):
        build_trusted_predicted_mask_registry_v1(artifact)


@pytest.mark.parametrize(
    "mutation",
    ["oracle", "missing_binding", "hash_mismatch", "caller_registry"],
)
def test_projector_returns_insufficient_evidence_for_untrusted_provenance(
    tmp_path, mutation
):
    from fluidmix.storage.artifacts.part_b_evidence import (
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
    from fluidmix.storage.artifacts.part_b_evidence import (
        PartBInsufficientEvidence,
        project_part_b_model_evidence_v1,
    )
    from fluidmix.perception.masks import build_trusted_predicted_mask_registry_v1

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
    from fluidmix.storage.artifacts.part_b_evidence import (
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
    from fluidmix.storage.artifacts.part_b_evidence import (
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
    from fluidmix.explanation.vlm.model_input_plan_builder import ModelInputPlanError, _validate_context

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


@pytest.mark.parametrize(
    ("module_name", "function_name", "arguments"),
    [
        (
            "evaluation.scoring.stage_cosmos_plan_artifact",
            "build_stage_attempt_plan_artifact",
            (None, None),
        ),
        (
            "evaluation.scoring.stage_cosmos_scoring",
            "score_stage_attempts",
            (None, None, None),
        ),
        (
            "evaluation.scoring.cosmos_plan_artifact",
            "build_v3_attempt_plan_artifact",
            (None, None),
        ),
        (
            "evaluation.scoring.cosmos_scoring",
            "score_v3_cosmos_attempts",
            (None, None, None),
        ),
    ],
)
def test_active_legacy_cosmos_entry_points_are_deactivated(
    module_name, function_name, arguments
):
    import importlib

    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    keyword = (
        {
            "decision_artifact_sha256": "a" * 64,
            "image_locator_sha256": "b" * 64,
        }
        if function_name == "build_v3_attempt_plan_artifact"
        else {}
    )
    with pytest.raises(RuntimeError, match="legacy_diagnostic_only"):
        function(*arguments, **keyword)


def test_active_cosmos_modules_do_not_import_legacy_serializers():
    forbidden = {
        "fluidmix.explanation.template.stage_cosmos_explanation",
        "fluidmix.explanation.template.varied_explanation",
    }
    for relative in (
        "src/evaluation/scoring/stage_cosmos_plan_artifact.py",
        "src/evaluation/scoring/stage_cosmos_scoring.py",
        "src/evaluation/scoring/cosmos_plan_artifact.py",
        "src/evaluation/scoring/cosmos_scoring.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert forbidden.isdisjoint(imported), relative


def test_part_boundary_manifest_registers_the_complete_current_boundary_inventory():
    from fluidmix.storage.artifacts.part_boundary_manifest import (
        EVALUATION_ONLY_MODULES,
        EVALUATION_SCRIPT_ENTRY_POINTS,
        LEGACY_ORACLE_DIAGNOSTIC_MODULES,
        PART_A_ONLY_MODULES,
        PRODUCTION_PART_B_ROOTS,
        PRODUCTION_SCRIPT_ENTRY_POINTS,
        SHARED_SUPPORT_MODULES,
    )

    assert {
        "twin.colour",
        "twin.io",
        "twin.paths",
        "twin.policy",
        "fluidmix.core.contracts.sim_metadata",
            "fluidmix.memory.reference_bank.materializer",
            "fluidmix.memory.reference_bank.postrender_producer",
            "fluidmix.memory.reference_bank.bank",
        "fluidmix.storage.artifacts.reference_bank_manifest",
        "twin.cohorts.reference_bank_manifest",
        "twin.scene.tank_colour_presets",
        "twin.physx.run_variation",
        "twin.transport.transform_attestation",
        "twin.transport.transport_stage",
        "twin.postrender.wp1_postrender",
    } <= PART_A_ONLY_MODULES
    assert {
        "fluidmix.pipeline.carryover_pipeline",
        "fluidmix.pipeline.history_population",
        "fluidmix.pipeline.sixth_development_run",
        "fluidmix.perception.cnn.cnn_predictor",
        "fluidmix.core.contracts.evidence",
        "fluidmix.core.contracts.mask_trust",
        "fluidmix.storage.sqlite.evidence_store",
        "fluidmix.storage.artifacts.part_b_evidence",
        "fluidmix.perception.materializer",
        "fluidmix.storage.sqlite.predicted_store_bridge",
        "fluidmix.perception.masks",
        "fluidmix.memory.history.sixth_development_run",
        "fluidmix.measurement.vision_measurement",
        "fluidmix.memory.history.development_reference",
        "fluidmix.pipeline.cohort_preparation",
        "fluidmix.memory.history.history",
        "fluidmix.pipeline.phase_capture",
        "fluidmix.perception.tracker.fill_step_tracker",
        "fluidmix.pipeline.tracker_receipt_adapter",
        "fluidmix.decision.vlm",
        "fluidmix.pipeline.service.accepted_adapter",
        "fluidmix.pipeline.service.http",
        "fluidmix.pipeline.service.manager",
    } <= PRODUCTION_PART_B_ROOTS
    assert {
        "fluidmix.storage.artifacts.capture_manifest_adapter",
        "fluidmix.perception.cnn.cnn_harness_recovery",
        "fluidmix.perception.cnn.cnn_holdout",
        "fluidmix.perception.cnn.cnn_training",
        "fluidmix.measurement.varied_stage_measurement",
            "fluidmix.pipeline.model_plan_measurement",
            "fluidmix.pipeline.model_input_plan",
            "fluidmix.pipeline.validation_stage_measurement",
            "fluidmix.pipeline.varied_stage_measurement",
        "fluidmix.decision.triage",
        "fluidmix.explanation.vlm.model_input_plan_builder",
        "fluidmix.memory.reference_bank.predicted_store_producer",
        "evaluation.rca_bench.rca_bench",
        "evaluation.rca_bench.rca_bench_controller",
        "evaluation.rca_bench.rca_bench_fatal",
        "fluidmix.decision.stage",
        "evaluation.scoring.synthetic_decision_scoring",
        "evaluation.scoring.stage_cosmos_scoring",
        "evaluation.scoring.cosmos_scoring",
        "fluidmix.measurement.fill_step_calibration",
        "fluidmix.explanation.vlm.development_materializer",
        "fluidmix.explanation.vlm.native_roi_rematerializer",
    } <= EVALUATION_ONLY_MODULES
    assert {
        "fluidmix.pipeline.service.contract",
        "fluidmix.pipeline.service.device",
        "fluidmix.pipeline.service.models",
        "fluidmix.pipeline.service.sources",
        "fluidmix.explanation.vlm.constrained",
        "fluidmix.explanation.vlm.image_support",
    } <= SHARED_SUPPORT_MODULES
    assert LEGACY_ORACLE_DIAGNOSTIC_MODULES == frozenset(
        {
            "fluidmix.explanation.template.varied_explanation",
            "fluidmix.explanation.template.stage_cosmos_explanation",
        }
    )
    assert PRODUCTION_SCRIPT_ENTRY_POINTS == frozenset(
        {
            "cli/fluidmix/perception/materialize_empty_reference_predictions.py",
            "cli/fluidmix/memory/history/materialize_sixth_development_run.py",
            "cli/fluidmix/storage/sqlite/populate_predicted_evidence_store.py",
            "cli/fluidmix/pipeline/prepare_wp7_cohort.py",
            "cli/fluidmix/memory/history/prepare_wp7_development_history.py",
            "cli/fluidmix/pipeline/run_carryover_pipeline.py",
        }
    )
    assert {
        "cli/fluidmix/explanation/vlm/build_model_input_plan_from_manifests.py",
        "cli/fluidmix/measurement/build_stage_measured_evidence.py",
        "cli/fluidmix/measurement/build_f2_varied_stage_measurements.py",
        "cli/fluidmix/decision/decide_f2_varied_triads.py",
    } <= EVALUATION_SCRIPT_ENTRY_POINTS
    assert {
        "cli/fluidmix/perception/cnn/audit_cnn_holdout_recovery.py",
        "cli/fluidmix/perception/cnn/build_cnn_training_inventory.py",
        "cli/fluidmix/perception/cnn/replay_cnn_holdout_bounds.py",
        "cli/fluidmix/perception/cnn/run_cnn_empty_control_once.py",
        "cli/fluidmix/perception/cnn/score_cnn_holdout_once.py",
        "cli/evaluation/scoring/score_synthetic_stage_decisions.py",
        "cli/fluidmix/perception/cnn/train_cnn_liquid_mask.py",
        "cli/evaluation/rca_bench/run_rca_bench.py",
        "cli/fluidmix/explanation/vlm/run_stage_cosmos_nano_eval.py",
        "cli/fluidmix/explanation/vlm/run_v3_cosmos_nano_eval.py",
        "cli/fluidmix/decision/aggregate_wp7_vlm_attempts.py",
        "cli/fluidmix/explanation/vlm/materialize_wp7_vlm_development_bundle.py",
        "cli/fluidmix/pipeline/prepare_wp7_development_reference_route.py",
        "cli/fluidmix/explanation/vlm/rematerialize_wp7_vlm_native_roi.py",
        "cli/fluidmix/explanation/vlm/run_wp7_vlm_model.py",
        "cli/fluidmix/decision/wp7_vlm_decision.py",
    } <= EVALUATION_SCRIPT_ENTRY_POINTS

    categories = (
        PART_A_ONLY_MODULES,
        PRODUCTION_PART_B_ROOTS,
        EVALUATION_ONLY_MODULES,
        LEGACY_ORACLE_DIAGNOSTIC_MODULES,
        SHARED_SUPPORT_MODULES,
    )
    for index, left in enumerate(categories):
        for right in categories[index + 1 :]:
            assert left.isdisjoint(right)

    current_modules = {
        f"{package}."
        + path.relative_to(ROOT / "src" / package)
        .with_suffix("")
        .as_posix()
        .replace("/", ".")
        for package in ("fluidmix", "twin", "evaluation")
        for path in (ROOT / "src" / package).rglob("*.py")
        if path.name != "__init__.py"
    }
    assert set().union(*categories) == current_modules


def test_local_import_parser_includes_constant_dynamic_imports():
    from fluidmix.storage.artifacts.part_boundary_manifest import parse_local_imports

    source = """
import fluidmix.decision.stage
import fluidmix.measurement.vision_measurement
from importlib import import_module as load_module
import importlib as importer

load_module("fluidmix.perception.masks")
importer.import_module("fluidmix.storage.artifacts.part_b_evidence")
__import__("fluidmix.explanation.vlm.model_input_plan_builder")
dynamic_name = "fluidmix.not_constant"
load_module(dynamic_name)
"""

    assert parse_local_imports(source, module_name="fluidmix.example") == frozenset(
        {
            "fluidmix.explanation.vlm.model_input_plan_builder",
            "fluidmix.storage.artifacts.part_b_evidence",
            "fluidmix.perception.masks",
            "fluidmix.decision.stage",
            "fluidmix.measurement.vision_measurement",
        }
    )
    assert parse_local_imports(
        "from . import evidence\n",
        module_name="fluidmix.core.contracts.example",
    ) == frozenset(
        {"fluidmix.core.contracts", "fluidmix.core.contracts.evidence"}
    )


def test_registered_production_and_active_evaluation_import_graph_is_isolated():
    from fluidmix.storage.artifacts.part_boundary_manifest import audit_current_dependency_isolation

    assert audit_current_dependency_isolation(ROOT) == ()


def test_live_service_production_graph_uses_only_registered_non_evaluation_modules():
    from fluidmix.storage.artifacts.part_boundary_manifest import (
        EVALUATION_ONLY_MODULES,
        PART_A_ONLY_MODULES,
        PRODUCTION_PART_B_ROOTS,
        walk_local_import_graph,
    )

    service_roots = {
        "fluidmix.pipeline.service.accepted_adapter",
        "fluidmix.pipeline.service.http",
        "fluidmix.pipeline.service.manager",
        "fluidmix.pipeline.tracker_receipt_adapter",
    }
    assert service_roots <= PRODUCTION_PART_B_ROOTS
    graph = walk_local_import_graph(ROOT, service_roots)
    reachable = set(graph).union(*(graph.values()))
    assert reachable.isdisjoint(PART_A_ONLY_MODULES | EVALUATION_ONLY_MODULES)
    assert (
        "fluidmix.wp7_vlm_image_support"
        in graph["fluidmix.pipeline.service.accepted_adapter"]
    )


def test_fresh_interpreter_forbidden_module_finder_guards_registered_roots():
    python = "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python"
    code = r"""
import importlib
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path.cwd() / "src"))

from fluidmix.storage.artifacts.part_boundary_manifest import (
    ACTIVE_EVALUATION_MODEL_SCRIPTS,
    LEGACY_ORACLE_DIAGNOSTIC_MODULES,
    PART_A_ONLY_MODULES,
    PRODUCTION_PART_B_ROOTS,
    PRODUCTION_SCRIPT_ENTRY_POINTS,
    install_forbidden_module_finder,
)

install_forbidden_module_finder(PART_A_ONLY_MODULES | LEGACY_ORACLE_DIAGNOSTIC_MODULES)
for name in sorted(PRODUCTION_PART_B_ROOTS):
    importlib.import_module(name)
for relative in sorted(PRODUCTION_SCRIPT_ENTRY_POINTS | ACTIVE_EVALUATION_MODEL_SCRIPTS):
    runpy.run_path(Path(relative), run_name="__part_boundary_import_smoke__")
try:
    importlib.import_module("fluidmix.core.contracts.sim_metadata")
except ModuleNotFoundError as error:
    assert "forbidden Part boundary import" in str(error)
else:
    raise AssertionError("forbidden-module finder did not reject Part A")
"""
    result = subprocess.run(
        [python, "-I", "-c", code],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fresh_process_read_guard_blocks_exact_part_a_targets_only(tmp_path):
    preset = tmp_path / "configs" / "tank_colour_presets.json"
    fixture = tmp_path / "fixtures" / "sim-metadata.json"
    receipt = tmp_path / "part-a-receipts" / "CASE-1" / "receipt.json"
    label = tmp_path / "part-a-labels" / "CASE-1" / "label.json"
    allowed_rgb = tmp_path / "camera" / "label-looking-rgb.png"
    allowed_recipe = tmp_path / "recipes" / "sim-metadata-recipe.json"
    for path in (preset, fixture, receipt, label, allowed_rgb, allowed_recipe):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name, encoding="utf-8")

    python = "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python"
    code = r"""
from pathlib import Path
import importlib
import os
import runpy
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
from fluidmix.storage.artifacts.part_boundary_manifest import (
    PRODUCTION_PART_B_ROOTS,
    PRODUCTION_SCRIPT_ENTRY_POINTS,
    install_production_part_b_read_guard,
)

paths = [Path(value) for value in sys.argv[1:]]
install_production_part_b_read_guard(
    paths[0].parents[1],
    sim_metadata_fixture_files=(paths[1],),
    part_a_receipt_roots=(paths[2].parents[1],),
    part_a_label_roots=(paths[3].parents[1],),
)
for name in sorted(PRODUCTION_PART_B_ROOTS):
    importlib.import_module(name)
for relative in sorted(PRODUCTION_SCRIPT_ENTRY_POINTS):
    runpy.run_path(Path(relative), run_name="__part_boundary_read_smoke__")
for path in paths[:4]:
    try:
        path.read_bytes()
    except PermissionError as error:
        assert "forbidden Part A read" in str(error)
    else:
        raise AssertionError(f"forbidden read succeeded: {path}")
try:
    descriptor = os.open(os.fsencode(paths[0]), os.O_RDONLY)
except PermissionError as error:
    assert "forbidden Part A read" in str(error)
else:
    os.close(descriptor)
    raise AssertionError("byte-path forbidden read succeeded")
assert paths[4].read_bytes()
assert paths[5].read_bytes()
"""
    result = subprocess.run(
        [
            python,
            "-I",
            "-c",
            code,
            *(str(path) for path in (
                preset,
                fixture,
                receipt,
                label,
                allowed_rgb,
                allowed_recipe,
            )),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_exact_read_guard_blocks_stable_derivation_reader_with_dir_fd(tmp_path):
    predecessor = tmp_path / "part-a" / "measured.json"
    derived = tmp_path / "part-b" / "decided.json"
    receipt = tmp_path / "part-b" / "decided.receipt.json"
    for path in (predecessor, derived, receipt):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    python = "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python"
    code = r"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
from fluidmix.core.contracts.evidence import verify_evidence_derivation_v1
from fluidmix.storage.artifacts.part_boundary_manifest import install_exact_read_guard

predecessor, derived, receipt = map(Path, sys.argv[1:])
install_exact_read_guard(exact_files=(predecessor,), forbidden_roots=())
try:
    verify_evidence_derivation_v1(
        predecessor,
        derived,
        receipt,
        predecessor_phase="measured",
        derived_phase="decided",
    )
except PermissionError as error:
    assert str(predecessor.resolve()) in str(error)
else:
    raise AssertionError("stable dir-fd reader bypassed the exact Part A guard")
"""
    result = subprocess.run(
        [python, "-I", "-c", code, str(predecessor), str(derived), str(receipt)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_palette_reel_root_is_classified_part_a_and_blocked_from_part_b_reads(tmp_path):
    from fluidmix.storage.artifacts.part_boundary_manifest import (
        PALETTE_CUT_REEL_ARTIFACT_TYPE,
        PART_A_ARTIFACT_TYPES,
        PART_A_REEL_ROOT_RELATIVE_PATH,
        validate_part_a_artifact_output,
    )

    classified = ROOT / PART_A_REEL_ROOT_RELATIVE_PATH
    classified.mkdir(parents=True, exist_ok=True)
    owned = classified / hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    owned.mkdir()
    output, root = validate_part_a_artifact_output(
        owned / "review-reel",
        repo_root=ROOT,
        artifact_type=PALETTE_CUT_REEL_ARTIFACT_TYPE,
    )
    assert PALETTE_CUT_REEL_ARTIFACT_TYPE in PART_A_ARTIFACT_TYPES
    assert output == owned / "review-reel"
    assert root == classified
    with pytest.raises(ValueError, match="unregistered Part A artifact type"):
        validate_part_a_artifact_output(
            owned / "wrong",
            repo_root=ROOT,
            artifact_type="unregistered_reel",
        )

    reel_file = owned / "review-reel/palette-cut-reel.json"
    reel_file.parent.mkdir()
    reel_file.write_text('{"artifact_type":"fluidmix_palette_cut_reel_v2"}\n')
    allowed = tmp_path / "part-b-input.json"
    allowed.write_text("{}\n")
    python = "/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python"
    code = r"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "src"))
from fluidmix.storage.artifacts.part_boundary_manifest import install_production_part_b_read_guard
reel, allowed = map(Path, sys.argv[1:])
install_production_part_b_read_guard(Path.cwd())
try:
    reel.read_bytes()
except PermissionError as error:
    assert "forbidden Part A read" in str(error)
else:
    raise AssertionError("classified Part-A reel read succeeded")
assert allowed.read_bytes() == b"{}\n"
"""
    completed = subprocess.run(
        [python, "-I", "-c", code, reel_file, allowed],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    shutil.rmtree(owned)


def test_post_inference_label_join_is_evaluation_only_and_requires_a_seal(tmp_path):
    from fluidmix.storage.artifacts.part_boundary_manifest import (
        LEGACY_ORACLE_DIAGNOSTIC_MODULES,
        POST_INFERENCE_LABEL_JOIN_ROOTS,
        PRODUCTION_PART_B_ROOTS,
        authorize_post_inference_label_join,
        seal_saved_inference_artifact,
        walk_local_import_graph,
    )

    assert POST_INFERENCE_LABEL_JOIN_ROOTS == frozenset(
        {
            "evaluation.scoring.empty_reference_development_scoring",
            "evaluation.scoring.synthetic_decision_scoring",
        }
    )
    graph = walk_local_import_graph(ROOT, PRODUCTION_PART_B_ROOTS)
    reachable = set().union(*(graph.values()), graph.keys())
    assert reachable.isdisjoint(POST_INFERENCE_LABEL_JOIN_ROOTS)
    assert reachable.isdisjoint(LEGACY_ORACLE_DIAGNOSTIC_MODULES)

    inference = tmp_path / "sealed-inference.json"
    inference.write_text('{"artifact_type":"measured_evidence"}\n', encoding="utf-8")
    digest = hashlib.sha256(inference.read_bytes()).hexdigest()
    seal = seal_saved_inference_artifact(
        inference,
        expected_sha256=digest,
        artifact_type="measured_evidence",
    )
    assert authorize_post_inference_label_join(
        "evaluation.scoring.synthetic_decision_scoring", seal
    ) is seal
    with pytest.raises(PermissionError, match="scoring-only"):
        authorize_post_inference_label_join("fluidmix.decision.stage", seal)
    with pytest.raises(TypeError, match="sealed inference"):
        authorize_post_inference_label_join(
            "evaluation.scoring.synthetic_decision_scoring", {"sha256": digest}
        )
    inference.write_text('{"artifact_type":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sealed inference artifact drifted"):
        authorize_post_inference_label_join(
            "evaluation.scoring.synthetic_decision_scoring", seal
        )


def test_wp7_future_roots_require_registration_and_the_production_projector():
    from fluidmix.storage.artifacts.part_boundary_manifest import (
        WP7_FUTURE_ROOTS,
        assert_wp7_dependency_policy,
        assert_wp7_root_registration,
    )

    assert WP7_FUTURE_ROOTS == frozenset({"fluidmix.pipeline.carryover_pipeline"})
    with pytest.raises(ValueError, match="unregistered WP7 root"):
        assert_wp7_root_registration(
            "fluidmix.wp7_future_inference",
            direct_dependencies={"fluidmix.storage.artifacts.part_b_evidence"},
        )
    with pytest.raises(ValueError, match="production projector"):
        assert_wp7_dependency_policy(set())
    with pytest.raises(ValueError, match="legacy oracle diagnostic"):
        assert_wp7_dependency_policy(
            {
                "fluidmix.storage.artifacts.part_b_evidence",
                "fluidmix.explanation.template.stage_cosmos_explanation",
            }
        )
    assert_wp7_dependency_policy({"fluidmix.storage.artifacts.part_b_evidence"})


def test_core_io_is_registered_as_shared_support():
    from fluidmix.storage.artifacts.part_boundary_manifest import module_ownership

    assert module_ownership("fluidmix.core.io") == "shared_support"


def test_authoritative_manifest_entries_resolve_and_unknown_roots_fail_closed():
    from fluidmix.storage.artifacts.part_boundary_manifest import (
        module_ownership,
        validate_boundary_manifest,
    )

    assert validate_boundary_manifest(ROOT) == ()
    assert module_ownership("fluidmix.storage.artifacts.part_b_evidence") == "production_part_b_root"
    assert module_ownership("fluidmix.core.contracts.sim_metadata") == "part_a_only"
    with pytest.raises(ValueError, match="unregistered boundary module"):
        module_ownership("fluidmix.future_unregistered_root")


def test_wp7_production_import_graph_excludes_the_evaluation_model_harness():
    from fluidmix.storage.artifacts.part_boundary_manifest import (
        EVALUATION_ONLY_MODULES,
        PRODUCTION_PART_B_ROOTS,
        walk_local_import_graph,
    )

    assert "fluidmix.wp7_vlm_decision" in PRODUCTION_PART_B_ROOTS
    graph = walk_local_import_graph(ROOT, {"fluidmix.decision.vlm"})
    reachable = set(graph).union(*(graph.values()))
    assert "fluidmix.model_bakeoff_runner" in EVALUATION_ONLY_MODULES
    assert reachable.isdisjoint(EVALUATION_ONLY_MODULES)

from __future__ import annotations

from copy import deepcopy
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCHEMA_ROOT = ROOT / "src" / "fluidmix" / "core" / "contracts"
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.core.contracts import (
    publish_evidence_derivation_v1,
    validate_evidence_record_v1,
    validate_evidence_sequence_v1,
    validate_sim_metadata_binding_v1,
    validate_sim_metadata_runtime_schedule_v1,
    validate_sim_metadata_v1,
    verify_evidence_artifacts,
    verify_evidence_derivation_v1,
    verify_sim_metadata_artifacts,
)
from twin.scene.tank_colour_presets import (
    load_tank_colour_preset_catalog,
    renderer_palette_from_resolved,
    resolve_tank_colour_preset,
    validate_tank_colour_resolution,
)
from twin.physx.dispense_residual import build_runtime_leading_residual_schedule
from twin.cohorts.one_run_manifest_v1 import build_one_run_source_artifacts

EVIDENCE_KEYS = {
    "run_id",
    "case_id",
    "vial_ordinal",
    "block_recipe_id",
    "block_position",
    "recipe_id",
    "previous_recipe_id",
    "recipe_ml",
    "dispense_order",
    "split_role",
    "split_version",
    "stage_id",
    "tank_id",
    "stage_ordinal",
    "fill_clip_path",
    "fill_frame_range",
    "transition_frame_ranges",
    "clip_sha256",
    "image_id",
    "rgb_sha256",
    "mask_sha256",
    "mask_source",
    "measurement_id",
    "median_lab",
    "mad_lab",
    "iqr_lab",
    "pairwise_delta_e76",
    "colour_regime_id",
    "evidence_id",
    "decision_id",
    "disposition",
    "limit_codes",
    "history_refs",
    "explanation_id",
    "model_revision",
    "prompt_sha256",
    "image_ids_sent",
    "guard_result",
}

SIM_KEYS = {
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


def _object_schemas(value: object):
    if isinstance(value, dict):
        if value.get("type") == "object" or "properties" in value:
            yield value
        for child in value.values():
            yield from _object_schemas(child)
    elif isinstance(value, list):
        for child in value:
            yield from _object_schemas(child)


def test_frozen_contract_schemas_use_draft_2020_12_and_close_every_object():
    for filename, exact_keys in (
        ("evidence-record-v1.schema.json", EVIDENCE_KEYS),
        ("sim-metadata-v1.schema.json", SIM_KEYS),
    ):
        schema = json.loads((SCHEMA_ROOT / filename).read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert set(schema["properties"]) == exact_keys
        assert set(schema["required"]) == exact_keys
        assert all(node.get("additionalProperties") is False for node in _object_schemas(schema))


def _measured_record(*, stage_ordinal: int = 1, tank_id: str = "t200") -> dict[str, object]:
    token = f"{stage_ordinal}-{tank_id}"
    return {
        "run_id": "RUN-1",
        "case_id": "CASE-1",
        "vial_ordinal": 1,
        "block_recipe_id": "RECIPE-1",
        "block_position": 1,
        "recipe_id": "RECIPE-1",
        "previous_recipe_id": "RECIPE-0",
        "recipe_ml": {"t100": 20.0, "t200": 10.0, "t400": 30.0},
        "dispense_order": ["t200", "t400", "t100"],
        "split_role": "train",
        "split_version": "whole-run-v1",
        "stage_id": f"STAGE-{token}",
        "tank_id": tank_id,
        "stage_ordinal": stage_ordinal,
        "fill_clip_path": "clips/RUN-1-CASE-1.mp4",
        "fill_frame_range": [10, 20],
        "transition_frame_ranges": {"entry": [5, 10], "exit": [20, 25]},
        "clip_sha256": "1" * 64,
        "image_id": f"IMAGE-{token}",
        "rgb_sha256": "2" * 64,
        "mask_sha256": "3" * 64,
        "mask_source": "oracle",
        "measurement_id": f"MEASUREMENT-{token}",
        "median_lab": [50.0, 1.0, -2.0],
        "mad_lab": [0.5, 0.25, 0.75],
        "iqr_lab": [1.0, 0.5, 1.5],
        "pairwise_delta_e76": [],
        "colour_regime_id": None,
        "evidence_id": f"EVIDENCE-{token}",
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


def test_measured_evidence_record_validates_and_returns_a_defensive_copy():
    record = _measured_record()
    validated = validate_evidence_record_v1(record, phase="measured", mode="storage")
    assert validated == record
    assert validated is not record


def test_measured_evidence_sequence_binds_identity_order_and_unique_stage_evidence():
    records = [
        _measured_record(stage_ordinal=1, tank_id="t200"),
        _measured_record(stage_ordinal=2, tank_id="t400"),
        _measured_record(stage_ordinal=3, tank_id="t100"),
    ]
    validated = validate_evidence_sequence_v1(
        records, phase="measured", mode="storage"
    )
    assert validated == records
    assert validated is not records
    assert all(actual is not source for actual, source in zip(validated, records, strict=True))


def test_oracle_evidence_artifacts_are_stable_read_and_exact_byte_hash_bound(tmp_path):
    clip = tmp_path / "clips" / "RUN-1-CASE-1.mp4"
    image = tmp_path / "images" / "IMAGE-1-t200.png"
    mask = tmp_path / "masks" / "oracle" / "IMAGE-1-t200.png"
    for path, payload in (
        (clip, b"clip-bytes"),
        (image, b"rgb-png-bytes"),
        (mask, b"mask-png-bytes"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    record = _measured_record()
    record["clip_sha256"] = hashlib.sha256(clip.read_bytes()).hexdigest()
    record["rgb_sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    record["mask_sha256"] = hashlib.sha256(mask.read_bytes()).hexdigest()
    verified = verify_evidence_artifacts(record, tmp_path)
    assert verified == record
    assert verified is not record


def test_evidence_derivation_publishes_and_verifies_exact_no_clobber_receipt(tmp_path):
    measured = _measured_record()
    decided = json.loads(json.dumps(measured))
    decided.update(
        {
            "decision_id": "DECISION-1",
            "disposition": "insufficient_evidence",
            "limit_codes": ["independent_reference_missing"],
            "history_refs": [
                {"evidence_id": "HISTORY-1", "query_sha256": "4" * 64}
            ],
        }
    )
    predecessor = tmp_path / "measured.json"
    derived = tmp_path / "decided.json"
    receipt = tmp_path / "decided.receipt.json"
    predecessor.write_text(
        json.dumps(measured, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    assert publish_evidence_derivation_v1(
        predecessor,
        derived,
        receipt,
        decided,
        predecessor_phase="measured",
        derived_phase="decided",
    ) == (derived, receipt)
    verify_evidence_derivation_v1(
        predecessor,
        derived,
        receipt,
        predecessor_phase="measured",
        derived_phase="decided",
    )
    first_record = derived.read_bytes()
    first_receipt = receipt.read_bytes()
    publish_evidence_derivation_v1(
        predecessor,
        derived,
        receipt,
        decided,
        predecessor_phase="measured",
        derived_phase="decided",
    )
    assert derived.read_bytes() == first_record
    assert receipt.read_bytes() == first_receipt


def test_tank_colour_resolution_validator_replays_task1_without_catalog_reads():
    resolved = resolve_tank_colour_preset(
        load_tank_colour_preset_catalog(),
        "amber_teal",
        transmission_overrides={"t200": [0.8, 0.5, 0.2]},
        dye_strength_scale=1.1,
        randomize_around_preset=True,
        randomization_seed=17,
        free_text_label="contract fixture",
    )
    record = resolved.to_json_record()
    validated = validate_tank_colour_resolution(record)
    assert validated.to_json_record() == record


def _sim_record() -> dict[str, object]:
    resolved = resolve_tank_colour_preset(
        load_tank_colour_preset_catalog(), "base"
    )
    resolution = resolved.to_json_record()
    renderer = renderer_palette_from_resolved(resolved)
    renderer.pop("excluded_from_model_inputs")
    advanced_keys = {
        "transmission_overrides",
        "dye_strength_scale",
        "t100_clear_lock",
        "randomize_around_preset",
        "randomization_seed",
        "tank_strength_jitter_bounds",
        "channel_jitter_bounds",
        "tank_strength_jitter",
        "channel_jitter",
        "free_text_label",
    }
    return {
        "tank_colour_preset": resolution["preset_name"],
        "resolved_tank_transmission_1cm": resolution["transmission_1cm"],
        "advanced_toggles": {
            key: resolution[key] for key in advanced_keys
        },
        "tank_colour_preset_hash": resolution["tank_colour_preset_hash"],
        "renderer_palette": renderer,
        "renderer_seed": 42,
        "residue_ml": 0.0,
        "residue_source_liquid": {
            "previous_recipe_id": None,
            "per_dispense": [],
        },
        "particle_state_summary": {
            "manifest_path": "particles/manifest.json",
            "manifest_sha256": "5" * 64,
            "manifest_bytes": 12,
            "particle_count": 3000,
            "finite": True,
            "resolution_profile_id": "f2-c21-v1",
        },
        "label": {
            "disposition": "keep",
            "rule_id": "prescribed_synthetic_latent_oracle",
        },
        "host_profile": {
            "profile_id": "GB10-LOCAL-V1",
            "compute_kind": "local",
            "gpu_model": "NVIDIA-GB10",
            "gpu_count": 1,
            "isaac_version": "6.2.0.0",
            "kit_version": "108.1",
            "profile_path": "hosts/gb10.json",
            "profile_sha256": "6" * 64,
            "profile_bytes": 10,
        },
        "excluded_from_model_inputs": True,
    }


def test_sim_metadata_validates_task1_resolution_and_residue_label_semantics():
    record = _sim_record()
    validated = validate_sim_metadata_v1(record)
    assert validated == record
    assert validated is not record


def test_sim_metadata_binding_receipt_binds_case_bytes_and_external_summaries(tmp_path):
    record = _sim_record()
    particle = tmp_path / "particles" / "manifest.json"
    host = tmp_path / "hosts" / "gb10.json"
    particle_payload = {
        "particle_count": 3000,
        "finite": True,
        "resolution_profile_id": "f2-c21-v1",
        "raw_payload_path": "particles/state.npz",
    }
    host_payload = {
        "profile_id": "GB10-LOCAL-V1",
        "compute_kind": "local",
        "gpu_model": "NVIDIA-GB10",
        "gpu_count": 1,
        "isaac_version": "6.2.0.0",
        "kit_version": "108.1",
        "driver_version": "external-only",
    }
    for path, payload in ((particle, particle_payload), (host, host_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    record["particle_state_summary"].update(  # type: ignore[union-attr]
        {
            "manifest_sha256": hashlib.sha256(particle.read_bytes()).hexdigest(),
            "manifest_bytes": particle.stat().st_size,
        }
    )
    record["host_profile"].update(  # type: ignore[union-attr]
        {
            "profile_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
            "profile_bytes": host.stat().st_size,
        }
    )
    metadata = tmp_path / "cases" / "CASE-1" / "sim-metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    binding = {
        "schema_version": "sim-metadata-binding-v1",
        "case_id": "CASE-1",
        "sim_metadata_path": "cases/CASE-1/sim-metadata.json",
        "sim_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    assert validate_sim_metadata_binding_v1(
        binding, expected_case_id="CASE-1", artifact_root=tmp_path
    ) == record


def test_sim_external_manifest_semantics_are_cross_checked_not_just_hashed(tmp_path):
    record = _sim_record()
    particle = tmp_path / "particles" / "manifest.json"
    host = tmp_path / "hosts" / "gb10.json"
    particle.parent.mkdir(parents=True)
    host.parent.mkdir(parents=True)
    particle.write_text(
        json.dumps(
            {
                "particle_count": 2999,
                "finite": True,
                "resolution_profile_id": "f2-c21-v1",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    host.write_text(
        json.dumps(
            {
                "profile_id": "GB10-LOCAL-V1",
                "compute_kind": "local",
                "gpu_model": "NVIDIA-GB10",
                "gpu_count": 1,
                "isaac_version": "6.2.0.0",
                "kit_version": "108.1",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    record["particle_state_summary"].update(  # type: ignore[union-attr]
        {
            "manifest_sha256": hashlib.sha256(particle.read_bytes()).hexdigest(),
            "manifest_bytes": particle.stat().st_size,
        }
    )
    record["host_profile"].update(  # type: ignore[union-attr]
        {
            "profile_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
            "profile_bytes": host.stat().st_size,
        }
    )
    with pytest.raises(ValueError, match="particle manifest content mismatch"):
        verify_sim_metadata_artifacts(record, tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("particle_count", True),
        ("finite", 1),
        ("gpu_count", True),
    ],
)
def test_sim_external_summaries_reject_bool_integer_type_confusion(
    tmp_path, field, invalid
):
    record = _sim_record()
    if field == "particle_count":
        record["particle_state_summary"]["particle_count"] = 1
    particle_payload = {
        "particle_count": record["particle_state_summary"]["particle_count"],
        "finite": True,
        "resolution_profile_id": "f2-c21-v1",
    }
    host_payload = {
        "profile_id": "GB10-LOCAL-V1",
        "compute_kind": "local",
        "gpu_model": "NVIDIA-GB10",
        "gpu_count": 1,
        "isaac_version": "6.2.0.0",
        "kit_version": "108.1",
    }
    target = particle_payload if field in particle_payload else host_payload
    target[field] = invalid
    particle = tmp_path / "particles" / "manifest.json"
    host = tmp_path / "hosts" / "gb10.json"
    for path, payload in ((particle, particle_payload), (host, host_payload)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    record["particle_state_summary"].update(
        {
            "manifest_sha256": hashlib.sha256(particle.read_bytes()).hexdigest(),
            "manifest_bytes": particle.stat().st_size,
        }
    )
    record["host_profile"].update(
        {
            "profile_sha256": hashlib.sha256(host.read_bytes()).hexdigest(),
            "profile_bytes": host.stat().st_size,
        }
    )
    with pytest.raises(ValueError, match="content mismatch"):
        verify_sim_metadata_artifacts(record, tmp_path)


def _positive_runtime_binding() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    schedule = build_runtime_leading_residual_schedule(
        current_recipe_id="RECIPE-1",
        previous_recipe_id="RECIPE-0",
        programmed_ml={"t100": 20.0, "t200": 10.0, "t400": 30.0},
        dispense_order=["t200", "t400", "t100"],
        previous_final_tank_id="t100",
        residual_volume_ml=1.0,
        duration_steps_by_tank={"t100": 120, "t200": 120, "t400": 120},
        physics_hz=120,
        capture_cadence_steps=4,
    )
    schedule["case_binding"] = {
        "case_id": "CASE-1",
        "residual_volume_ml": 1.0,
        "previous_recipe_id": "RECIPE-0",
    }
    record = _sim_record()
    record["residue_ml"] = 1.0
    record["residue_source_liquid"] = {
        "previous_recipe_id": "RECIPE-0",
        "per_dispense": [
            {
                "stage_ordinal": pour["ordinal"],
                "source_tank_id": pour["residual_source_tank_id"],
                "composition_t200_t400_t100": list(
                    pour["emission_segments"][0]["one_hot_composition"]
                ),
            }
            for pour in schedule["pours"]
        ],
    }
    record["label"] = {
        "disposition": "discard",
        "rule_id": "prescribed_synthetic_latent_oracle",
    }
    schedule_hash = hashlib.sha256(
        json.dumps(
            schedule, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    label = {
        "case_id": "CASE-1",
        "prescribed_leading_residual_volume_ml": 1.0,
        "expected_disposition": "discard",
        "label_source": "prescribed_synthetic_latent_oracle",
        "render_schedule_sha256": schedule_hash,
        "excluded_from_model_inputs": True,
        "excluded_from_deploy_inputs": True,
    }
    return record, schedule, label


def test_sim_metadata_runtime_binding_matches_per_dispense_sources_and_label():
    record, schedule, label = _positive_runtime_binding()
    validate_sim_metadata_runtime_schedule_v1(
        record, schedule, label, expected_case_id="CASE-1"
    )

    drifted = deepcopy(schedule)
    drifted["case_binding"]["previous_recipe_id"] = "OTHER-RECIPE"
    label["render_schedule_sha256"] = hashlib.sha256(
        json.dumps(
            drifted, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="previous_recipe_id mismatch"):
        validate_sim_metadata_runtime_schedule_v1(
            record, drifted, label, expected_case_id="CASE-1"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_version_bool",
        "label_residue_bool",
        "schedule_residue_bool",
        "case_residue_bool",
        "pour_residue_bool",
        "pour_ordinal_bool",
        "composition_bool",
        "model_previous_final_tank_drift",
        "dispense_tank_bool",
        "dispense_tank_duplicate",
        "current_segment_mismatch",
    ],
)
def test_runtime_binding_rejects_bool_numeric_and_source_context_drift(mutation):
    record, schedule, label = _positive_runtime_binding()
    if mutation == "schema_version_bool":
        schedule["schema_version"] = True
    elif mutation == "label_residue_bool":
        label["prescribed_leading_residual_volume_ml"] = True
    elif mutation == "schedule_residue_bool":
        schedule["label_provenance"]["leading_residual_volume_ml"] = True
    elif mutation == "case_residue_bool":
        schedule["case_binding"]["residual_volume_ml"] = True
    elif mutation == "pour_residue_bool":
        schedule["pours"][0]["leading_residual_volume_ml"] = True
    elif mutation == "pour_ordinal_bool":
        schedule["pours"][0]["ordinal"] = True
    elif mutation == "composition_bool":
        composition = schedule["pours"][0]["emission_segments"][0][
            "one_hot_composition"
        ]
        composition[composition.index(1.0)] = True
    elif mutation == "model_previous_final_tank_drift":
        schedule["model_input_record"]["previous_final_tank_id"] = "t200"
    elif mutation == "current_segment_mismatch":
        schedule["pours"][0]["emission_segments"][1]["source_tank_id"] = "t100"
    else:
        invalid_tank = True if mutation == "dispense_tank_bool" else "t400"
        schedule["physical_sequence"]["dispense_order"][-1] = invalid_tank
        schedule["model_input_record"]["dispense_order"][-1] = invalid_tank
        schedule["pours"][-1]["tank_id"] = invalid_tank
    label["render_schedule_sha256"] = hashlib.sha256(
        json.dumps(
            schedule, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_sim_metadata_runtime_schedule_v1(
            record, schedule, label, expected_case_id="CASE-1"
        )


def test_evidence_record_phase_range_pair_and_production_rules_fail_closed():
    record = _measured_record()
    record["fill_clip_path"] = "/tmp/outside.mp4"
    with pytest.raises(ValueError, match="contained relative POSIX"):
        validate_evidence_record_v1(record, phase="measured")

    record = _measured_record()
    record["fill_clip_path"] = "."
    with pytest.raises(ValueError, match="contained relative POSIX"):
        validate_evidence_record_v1(record, phase="measured")

    record = _measured_record()
    record["transition_frame_ranges"]["entry"] = [5, 11]  # type: ignore[index]
    with pytest.raises(ValueError, match="overlaps"):
        validate_evidence_record_v1(record, phase="measured")

    record = _measured_record()
    record["pairwise_delta_e76"] = [
        {"other_evidence_id": record["evidence_id"], "delta_e76": 0.0}
    ]
    with pytest.raises(ValueError, match="self-pair"):
        validate_evidence_record_v1(record, phase="measured")

    record = _measured_record()
    record["decision_id"] = "DECISION-TOO-EARLY"
    with pytest.raises(ValueError, match="measured records"):
        validate_evidence_record_v1(record, phase="measured")

    with pytest.raises(ValueError, match="oracle"):
        validate_evidence_record_v1(
            _measured_record(), phase="measured", mode="production"
        )
    predicted = _measured_record()
    predicted["mask_source"] = "predicted"
    validate_evidence_record_v1(predicted, phase="measured", mode="production")

    regime = _measured_record()
    regime["colour_regime_id"] = "PRESET-ALIAS"
    with pytest.raises(ValueError, match="WP5"):
        validate_evidence_record_v1(regime, phase="measured")
    with pytest.raises(ValueError, match="WP5"):
        validate_evidence_record_v1(
            _measured_record(),
            phase="measured",
            verified_colour_regime_binding=object(),
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("image_id", "IMAGE-1\n"),
        ("run_id", "RUN\u202e-1"),
        ("evidence_id", "EVIDENCE\u0085-1"),
        ("clip_sha256", "a" * 64 + "\n"),
    ],
)
def test_evidence_ids_and_hashes_reject_trailing_or_unicode_controls(field, invalid):
    record = _measured_record()
    record[field] = invalid
    with pytest.raises(ValueError):
        validate_evidence_record_v1(record, phase="measured")


def test_sim_metadata_ids_reject_unicode_controls():
    record = _sim_record()
    record["host_profile"]["profile_id"] = "GB10\u202e-LOCAL"  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_sim_metadata_v1(record)


def test_decided_visual_and_template_explained_phase_shapes_are_exact():
    decided = _measured_record()
    decided.update(
        {
            "decision_id": "DECISION-1",
            "disposition": "keep",
            "limit_codes": ["synthetic_only"],
        }
    )
    validate_evidence_record_v1(decided, phase="decided")

    visual = deepcopy(decided)
    visual.update(
        {
            "explanation_id": "EXPLANATION-1",
            "model_revision": "open-vlm-revision-1",
            "prompt_sha256": "7" * 64,
            "image_ids_sent": ["IMAGE-A", "IMAGE-B", "IMAGE-C", "IMAGE-D"],
            "guard_result": "pass",
        }
    )
    validate_evidence_record_v1(visual, phase="explained")
    visual["image_ids_sent"] = ["IMAGE-A"]
    with pytest.raises(ValueError, match="4 or 6"):
        validate_evidence_record_v1(visual, phase="explained")

    template = deepcopy(decided)
    template.update(
        {
            "explanation_id": "EXPLANATION-TEMPLATE-1",
            "model_revision": "deterministic-template:v1",
            "prompt_sha256": "8" * 64,
            "image_ids_sent": [],
            "guard_result": "fallback",
        }
    )
    validate_evidence_record_v1(template, phase="explained")


def test_evidence_sequence_rejects_identity_duplicate_and_tank_order_drift():
    records = [
        _measured_record(stage_ordinal=1, tank_id="t200"),
        _measured_record(stage_ordinal=2, tank_id="t400"),
        _measured_record(stage_ordinal=3, tank_id="t100"),
    ]
    changed = deepcopy(records)
    changed[1]["run_id"] = "OTHER-RUN"
    with pytest.raises(ValueError, match="shared run_id"):
        validate_evidence_sequence_v1(changed, phase="measured")
    changed = deepcopy(records)
    changed[1]["evidence_id"] = changed[0]["evidence_id"]
    with pytest.raises(ValueError, match="unique evidence_id"):
        validate_evidence_sequence_v1(changed, phase="measured")
    changed = deepcopy(records)
    changed[1]["tank_id"], changed[2]["tank_id"] = (
        changed[2]["tank_id"],
        changed[1]["tank_id"],
    )
    with pytest.raises(ValueError, match="dispense_order"):
        validate_evidence_sequence_v1(changed, phase="measured")


def test_predicted_storage_registry_uses_exact_bound_paths_and_rejects_symlinks(tmp_path):
    clip = tmp_path / "clips" / "RUN-1-CASE-1.mp4"
    rgb = tmp_path / "predicted" / "rgb.png"
    mask = tmp_path / "predicted" / "mask.png"
    for path, raw in ((clip, b"clip"), (rgb, b"rgb"), (mask, b"mask")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    record = _measured_record()
    record.update(
        {
            "mask_source": "predicted",
            "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
            "rgb_sha256": hashlib.sha256(rgb.read_bytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
        }
    )
    locator = {
        "rgb_path": str(rgb),
        "rgb_sha256": record["rgb_sha256"],
        "mask_path": str(mask),
        "mask_sha256": record["mask_sha256"],
    }
    registry = {record["image_id"]: locator}
    verify_evidence_artifacts(record, tmp_path, predicted_mask_registry=registry)

    extra = deepcopy(registry)
    extra[record["image_id"]]["producer"] = "caller-asserted"  # type: ignore[index]
    with pytest.raises(ValueError, match="invalid fields"):
        verify_evidence_artifacts(record, tmp_path, predicted_mask_registry=extra)

    linked = tmp_path / "predicted" / "linked-mask.png"
    linked.symlink_to(mask)
    symlinked = deepcopy(registry)
    symlinked[record["image_id"]]["mask_path"] = str(linked)  # type: ignore[index]
    with pytest.raises(ValueError, match="unsafe"):
        verify_evidence_artifacts(record, tmp_path, predicted_mask_registry=symlinked)


def test_derivation_rejects_nonowning_change_and_recovers_record_first_orphan(tmp_path):
    measured = _measured_record()
    decided = deepcopy(measured)
    decided.update({"decision_id": "D-1", "disposition": "keep"})
    predecessor = tmp_path / "measured.json"
    derived = tmp_path / "decided.json"
    receipt = tmp_path / "receipt.json"
    predecessor.write_text(
        json.dumps(measured, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    derived.write_text(
        json.dumps(decided, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    publish_evidence_derivation_v1(
        predecessor,
        derived,
        receipt,
        decided,
        predecessor_phase="measured",
        derived_phase="decided",
    )
    assert receipt.is_file()

    invalid = deepcopy(decided)
    invalid["recipe_id"] = "MUTATED-RECIPE"
    other_derived = tmp_path / "invalid.json"
    other_receipt = tmp_path / "invalid.receipt.json"
    with pytest.raises(ValueError, match="owning phase"):
        publish_evidence_derivation_v1(
            predecessor,
            other_derived,
            other_receipt,
            invalid,
            predecessor_phase="measured",
            derived_phase="decided",
        )


def test_task1_validator_rejects_self_consistent_but_invented_resolution():
    resolved = resolve_tank_colour_preset(
        load_tank_colour_preset_catalog(), "base"
    ).to_json_record()
    forged = deepcopy(resolved)
    forged["transmission_1cm"]["t200"][0] = 0.8  # type: ignore[index]
    renderer_hash = hashlib.sha256(
        json.dumps(
            forged["transmission_1cm"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    forged["renderer_palette_sha256"] = renderer_hash
    forged["palette_sha256"] = renderer_hash
    semantic = {
        key: value
        for key, value in forged.items()
        if key
        not in {
            "tank_colour_preset_hash",
            "renderer_palette_sha256",
            "palette_sha256",
        }
    }
    forged["tank_colour_preset_hash"] = hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="Task 1 semantics"):
        validate_tank_colour_resolution(forged)


def test_task1_validator_rejects_canonical_json_numeric_type_drift():
    record = resolve_tank_colour_preset(
        load_tank_colour_preset_catalog(), "base"
    ).to_json_record()
    record["transmission_1cm"]["t100"][0] = 1
    with pytest.raises(ValueError, match="Task 1 semantics"):
        validate_tank_colour_resolution(record)


@pytest.mark.parametrize("drift", ["root_and_renderer", "renderer_only"])
def test_sim_metadata_rejects_canonical_palette_numeric_type_drift(drift):
    record = _sim_record()
    if drift == "root_and_renderer":
        record["resolved_tank_transmission_1cm"]["t100"][0] = 1
    record["renderer_palette"]["transmission_1cm"]["t100"][0] = 1
    with pytest.raises(ValueError):
        validate_sim_metadata_v1(record)


def test_sim_metadata_rejects_label_source_and_runtime_schedule_drift():
    label_drift = _sim_record()
    label_drift["label"]["disposition"] = "discard"  # type: ignore[index]
    with pytest.raises(ValueError, match="contradicts"):
        validate_sim_metadata_v1(label_drift)

    schedule = build_runtime_leading_residual_schedule(
        current_recipe_id="RECIPE-1",
        previous_recipe_id="RECIPE-0",
        programmed_ml={"t100": 20.0, "t200": 10.0, "t400": 30.0},
        dispense_order=["t200", "t400", "t100"],
        previous_final_tank_id="t100",
        residual_volume_ml=0.0,
        duration_steps_by_tank={"t100": 120, "t200": 120, "t400": 120},
        physics_hz=120,
        capture_cadence_steps=4,
    )
    schedule["case_binding"] = {
        "case_id": "CASE-1",
        "residual_volume_ml": 0.0,
        "previous_recipe_id": "RECIPE-0",
    }
    schedule["artifact_type"] = "physx_f2_leading_residual_schedule"
    label = {
        "case_id": "CASE-1",
        "prescribed_leading_residual_volume_ml": 0.0,
        "expected_disposition": "keep",
        "label_source": "prescribed_synthetic_latent_oracle",
        "render_schedule_sha256": "0" * 64,
        "excluded_from_model_inputs": True,
        "excluded_from_deploy_inputs": True,
    }
    with pytest.raises(ValueError, match="finalized runtime"):
        validate_sim_metadata_runtime_schedule_v1(
            _sim_record(), schedule, label, expected_case_id="CASE-1"
        )


def test_zero_residue_runtime_binding_has_no_fictional_source_recipe():
    schedule = build_runtime_leading_residual_schedule(
        current_recipe_id="RECIPE-1",
        previous_recipe_id="RECIPE-0",
        programmed_ml={"t100": 20.0, "t200": 10.0, "t400": 30.0},
        dispense_order=["t200", "t400", "t100"],
        previous_final_tank_id="t100",
        residual_volume_ml=0.0,
        duration_steps_by_tank={"t100": 120, "t200": 120, "t400": 120},
        physics_hz=120,
        capture_cadence_steps=4,
    )
    schedule["case_binding"] = {
        "case_id": "CASE-1",
        "residual_volume_ml": 0.0,
        "previous_recipe_id": "RECIPE-0",
    }
    label = {
        "case_id": "CASE-1",
        "prescribed_leading_residual_volume_ml": 0.0,
        "expected_disposition": "keep",
        "label_source": "prescribed_synthetic_latent_oracle",
        "render_schedule_sha256": hashlib.sha256(
            json.dumps(
                schedule, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest(),
        "excluded_from_model_inputs": True,
        "excluded_from_deploy_inputs": True,
    }
    validate_sim_metadata_runtime_schedule_v1(
        _sim_record(), schedule, label, expected_case_id="CASE-1"
    )

    invalid = deepcopy(schedule)
    invalid["pours"][0]["ordinal"] = True
    label["render_schedule_sha256"] = hashlib.sha256(
        json.dumps(
            invalid, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_sim_metadata_runtime_schedule_v1(
            _sim_record(), invalid, label, expected_case_id="CASE-1"
        )


def test_runtime_binding_accepts_approved_task2_clean_start_context():
    artifacts = build_one_run_source_artifacts(
        {
            "schema_version": 1,
            "artifact_type": "fluidmix-one-run-source-spec-v1",
            "run_id": "RUN-CLEAN-START-1",
            "split_role": "train",
            "split_version": "dynamic-one-run-v1",
            "source_sha256": {
                "recipe_manifest": "a" * 64,
                "split_spec": "b" * 64,
            },
            "recipes": [
                {
                    "recipe_id": "A",
                    "programmed_ml": {"t100": 20.0, "t200": 20.0, "t400": 20.0},
                },
                {
                    "recipe_id": "B",
                    "programmed_ml": {"t100": 10.0, "t200": 30.0, "t400": 20.0},
                },
            ],
            "initial_condition": "clean_start",
            "initial_predecessor_recipe_id": None,
            "vial_recipe_ids": ["A", "A", "B"],
            "residual_volume_ml_by_vial": [0.0, 0.0, 9.0],
            "render_seed_by_vial": [720001, 720002, 720003],
            "nuisance_group_id_by_vial": ["CLEAN-1", "CLEAN-2", "CLEAN-3"],
        }
    )
    schedule = artifacts["render-schedule-plan.json"]["execution_schedules"][0]
    label = artifacts["label-provenance.json"]["vial_labels"][0]
    case_id = schedule["case_binding"]["case_id"]
    assert schedule["case_binding"]["previous_recipe_id"] is None
    assert schedule["model_input_record"]["previous_final_tank_id"] is None
    validate_sim_metadata_runtime_schedule_v1(
        _sim_record(), schedule, label, expected_case_id=case_id
    )

    permuted = deepcopy(schedule)
    order = list(reversed(permuted["physical_sequence"]["dispense_order"]))
    permuted["physical_sequence"]["dispense_order"] = list(order)
    permuted["model_input_record"]["dispense_order"] = list(order)
    for pour, tank_id in zip(permuted["pours"], order, strict=True):
        pour["tank_id"] = tank_id
    invalid_label = deepcopy(label)
    invalid_label["render_schedule_sha256"] = hashlib.sha256(
        json.dumps(
            permuted, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_sim_metadata_runtime_schedule_v1(
            _sim_record(), permuted, invalid_label, expected_case_id=case_id
        )

    for owner, field in (
        ("case_binding", "previous_recipe_id"),
        ("model_input_record", "previous_recipe_id"),
        ("model_input_record", "previous_final_tank_id"),
    ):
        missing = deepcopy(schedule)
        del missing[owner][field]
        invalid_label = deepcopy(label)
        invalid_label["render_schedule_sha256"] = hashlib.sha256(
            json.dumps(
                missing, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
        with pytest.raises(ValueError):
            validate_sim_metadata_runtime_schedule_v1(
                _sim_record(), missing, invalid_label, expected_case_id=case_id
            )


def test_contract_package_keeps_evidence_import_isolated_from_part_a():
    script = """
import sys
import fluidmix.core.contracts as contracts
assert 'fluidmix.contracts.sim_metadata' not in sys.modules
assert 'fluidmix.tank_colour_presets' not in sys.modules
getattr(contracts, 'validate_evidence_record_v1')
assert 'fluidmix.contracts.sim_metadata' not in sys.modules
assert 'fluidmix.tank_colour_presets' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

"""Public-boundary tests for the WP7 final-decision production wiring."""

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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_capture_fixture(
    tmp_path: Path, recipe: dict[str, object], video: Path
) -> tuple[Path, str]:
    from test_carryover_pipeline import _write_json

    frames = [
        np.full((480, 640, 3), index, dtype=np.uint8)
        for index in range(16)
    ]
    fills = [
        {"case_id": f"CASE-{index}", "frame_count": 2}
        for index in range(1, 5)
    ]
    transitions = [
        {"segment_index": index, "start_frame_index": start,
         "stop_frame_index": stop, "frame_count": stop - start,
         "decoded_bgr_sha256": None}
        for index, (start, stop) in enumerate(
            ((0, 2), (3, 5), (6, 10), (11, 13), (14, 16))
        )
    ]
    fill_windows = [
        {
            "fill_index": index,
            "case_id": fill["case_id"],
            "start_frame_index": transitions[index]["stop_frame_index"] - 1,
            "stop_frame_index": transitions[index + 1]["start_frame_index"],
            "frame_count": fill["frame_count"],
        }
        for index, fill in enumerate(fills)
    ]
    timing_sha = _canonical_hash({
        "fill_clips": fills,
        "transition_ranges": [
            {key: row[key] for key in (
                "segment_index", "start_frame_index", "stop_frame_index",
                "frame_count",
            )}
            for row in transitions
        ],
        "fill_windows": fill_windows,
    })
    run_bindings = []
    run_transitions = [
        {"segment_index": 0, "start_frame_index": 0,
         "stop_frame_index": 2, "frame_count": 2,
         "decoded_bgr_sha256": []},
        {"segment_index": 1, "start_frame_index": 3,
         "stop_frame_index": 5, "frame_count": 2,
         "decoded_bgr_sha256": []},
        {"segment_index": 2, "start_frame_index": 6,
         "stop_frame_index": 8, "frame_count": 2,
         "decoded_bgr_sha256": []},
    ]
    for index, run_fills in enumerate((fills[:2], fills[2:]), start=1):
        run_path = tmp_path / f"capture-source-run-{index}.json"
        run_video_sha = f"{index:064x}"
        _write_json(run_path, {
            "artifact_type": "rendered_transport_run_sequence_v4",
            "status": "pass", "run_id": f"RUN-{index}",
            "case_ids": [row["case_id"] for row in run_fills],
            "output_video_sha256": run_video_sha,
            "output_frame_count": 8,
            "encoded_transition_evidence": run_transitions,
            "fill_clips": run_fills,
        })
        run_bindings.append({
            "path": str(run_path.absolute()), "sha256": _sha(run_path),
            "artifact_type": "rendered_transport_run_sequence_v4",
            "run_id": f"RUN-{index}",
            "case_ids": [row["case_id"] for row in run_fills],
            "output_video_sha256": run_video_sha,
            "output_frame_count": 8,
        })

    video_sha = _sha(video)
    recipe_sha = _canonical_hash(recipe)
    sources = []
    composition_sha = None
    upstream_values = (
        (
            "ordered_wp1_composition_receipt",
            "wp7_ordered_test_video_v1",
            {
                "schema_version": 1,
                "artifact_type": "wp7_ordered_test_video_v1",
                "status": "pass",
                "output": {"sha256": video_sha, "frame_count": len(frames)},
                "ordered_runs": [
                    {
                        "run_id": row["run_id"], "case_ids": row["case_ids"],
                        "run_receipt_path": row["path"],
                        "run_receipt_sha256": row["sha256"],
                        "video_sha256": row["output_video_sha256"],
                        "frame_count": row["output_frame_count"],
                    }
                    for row in run_bindings
                ],
                "verification": {
                    "case_count": 4, "run_count": 2,
                    "source_order": ["RUN-1", "RUN-2"],
                },
            },
        ),
        (
            "ordered_wp1_review_receipt",
            "fluidmix_wp7_ordered_test_video_final_output_review_v1",
            None,
        ),
        (
            "ordered_wp1_run_receipt",
            "wp7_ordered_test_run_receipt_v1",
            {
                "schema_version": 1,
                "artifact_type": "wp7_ordered_test_run_receipt_v1",
                "status": "pass", "run_receipts": run_bindings,
            },
        ),
    )
    for role, artifact_type, initial_value in upstream_values:
        upstream_path = tmp_path / f"capture-upstream-{role}.json"
        if role == "ordered_wp1_review_receipt":
            upstream_value = {
                "schema_version": 1, "artifact_type": artifact_type,
                "status": "pass", "verdict": "pass",
                "identity": {
                    "output_mp4_sha256": video_sha,
                    "output_receipt_sha256": composition_sha,
                },
                "case_contract": {"count": 4, "unique": True},
                "output_media": {"sha256": video_sha, "frame_count": len(frames)},
            }
        else:
            upstream_value = initial_value
        _write_json(upstream_path, upstream_value)
        if role == "ordered_wp1_composition_receipt":
            composition_sha = _sha(upstream_path)
        adapter_path = tmp_path / f"capture-adapter-{role}.json"
        _write_json(adapter_path, {
            "schema_version": "wp7-capture-source-adapter-v1",
            "artifact_type": "wp7_capture_source_adapter",
            "phase": "run-test", "role": role,
            "video_sha256": video_sha,
            "recipe_sequence_sha256": recipe_sha,
            "ordered_case_ids": [row["case_id"] for row in fills],
            "capture_timing_sha256": timing_sha,
            "upstream_artifact_type": artifact_type,
            "upstream_path": str(upstream_path.absolute()),
            "upstream_sha256": _sha(upstream_path), "status": "pass",
        })
        sources.append({
            "role": role, "path": str(adapter_path.absolute()),
            "sha256": _sha(adapter_path),
        })

    plan_path = tmp_path / "capture-plan.json"
    _write_json(plan_path, {
        "schema_version": "wp7-phase-capture-plan-v1",
        "artifact_type": "wp7_phase_capture_plan",
        "phase": "run-test", "capture_origin": "synthetic_camera",
        "video_sha256": video_sha, "fill_clips": fills,
        "encoded_transition_evidence": transitions,
        "source_provenance": sources,
    })
    return plan_path, _sha(plan_path)


def _capture_loader(kwargs: dict[str, object], runtime):
    from fluidmix.wp7_phase_capture import publish_phase_capture

    def load(video_bytes, recipe, phase, video_sha256):
        assert hashlib.sha256(video_bytes).hexdigest() == video_sha256
        publish_phase_capture(
            plan_path=kwargs["capture_plan"],
            expected_plan_sha256=kwargs["capture_plan_sha256"],
            expected_phase=phase,
            expected_capture_origin="synthetic_camera",
            video_bytes=video_bytes,
            recipe_sequence=recipe,
            decoded_rgb_frames=[
                np.full((480, 640, 3), index, dtype=np.uint8)
                for index in range(16)
            ],
            capture_receipt_out=kwargs["capture_receipt_out"],
            provenance_out=kwargs["capture_provenance_out"],
        )
        return runtime

    return load


def _runtime_output(tmp_path: Path, recipe: dict[str, object]):
    import fluidmix.vision_measurement as measurement
    from fluidmix.carryover_pipeline import RuntimeProjectionInput
    from fluidmix.rgb_mask_prediction import build_trusted_predicted_mask_registry_v1

    images = {}
    templates = []
    locators = {}
    cursors = {}
    bindings = []
    video = tmp_path / "video.bin"
    cursor = 0
    selected_frames = {}
    measurement_citations = {}
    for run in recipe["runs"]:
        for vial in run["vials"]:
            for ordinal, tank in enumerate(vial["dispense_order"], start=1):
                frame_index = vial["vial_ordinal"] * 100 + ordinal
                image_id = f"{vial['case_id']}-F{frame_index:06d}"
                selected_frames.setdefault(vial["case_id"], []).append(frame_index)
                rgb = tmp_path / "rgb" / f"{image_id}.png"
                mask = tmp_path / "mask" / f"{image_id}.png"
                rgb.parent.mkdir(exist_ok=True)
                mask.parent.mkdir(exist_ok=True)
                colour = (80, 100, 120) if vial["vial_ordinal"] != 2 else (220, 20, 20)
                Image.new("RGB", (4, 3), colour).save(rgb)
                pixels = np.zeros((3, 4), dtype=np.uint8)
                if vial["vial_ordinal"] != 4:
                    pixels[1:, 1:3] = 255
                Image.fromarray(pixels, mode="L").save(mask)
                images[image_id] = {
                    "rgb_path": str(rgb), "rgb_sha256": _sha(rgb),
                    "mask_path": str(mask), "mask_sha256": _sha(mask),
                }
    artifact = {
        "schema_version": 2, "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin", "mask_source": "rgb_only_prediction",
        "producer": {
            "producer_id": "segmenter-1", "producer_kind": "rgb_only_liquid_segmenter",
            "model_id": "model-1", "model_sha256": "a" * 64,
            "config_id": "config-1", "config_sha256": "b" * 64,
            "code_commit": "c" * 40, "threshold_source": "twin_validation_frozen",
            "training_data_origin": "twin", "input_fields": ["rgb"],
        },
        "images": images,
    }
    registry = build_trusted_predicted_mask_registry_v1(artifact)
    for run in recipe["runs"]:
        for vial in run["vials"]:
            for ordinal, tank in enumerate(vial["dispense_order"], start=1):
                frame_index = vial["vial_ordinal"] * 100 + ordinal
                image_id = f"{vial['case_id']}-F{frame_index:06d}"
                evidence_id = f"EVID-{vial['case_id']}-{tank}"
                measured, binding = measurement.measure_trusted_predicted_image(
                    registry.entry(image_id),
                    measurement_id=f"MEAS-{vial['case_id']}-{tank}",
                    evidence_id=evidence_id,
                )
                bindings.append(binding)
                measurement_citations.setdefault(vial["case_id"], []).append(
                    binding.measurement_sha256
                )
                entry = registry.entry(image_id)
                templates.append({
                    "run_id": run["run_id"], "case_id": vial["case_id"],
                    "vial_ordinal": vial["vial_ordinal"],
                    "block_recipe_id": vial["block_recipe_id"],
                    "block_position": vial["block_position"],
                    "recipe_id": vial["recipe_id"],
                    "previous_recipe_id": vial["previous_recipe_id"],
                    "recipe_ml": dict(vial["programmed_tank_ml"]),
                    "dispense_order": list(vial["dispense_order"]),
                    "split_role": recipe["split_role"],
                    "split_version": recipe["split_version"],
                    "stage_id": f"STAGE-{vial['case_id']}-{ordinal}",
                    "tank_id": tank, "stage_ordinal": ordinal,
                    "fill_clip_path": video.name, "fill_frame_range": [0, 1],
                    "transition_frame_ranges": {"entry": None, "exit": None},
                    "clip_sha256": _sha(video), "image_id": image_id,
                    "rgb_sha256": entry.rgb_sha256, "mask_sha256": entry.mask_sha256,
                    "mask_source": "predicted", "measurement_id": binding.measurement_id,
                    "median_lab": [0.0, 0.0, 0.0], "mad_lab": [0.0, 0.0, 0.0],
                    "iqr_lab": [0.0, 0.0, 0.0], "pairwise_delta_e76": [],
                    "colour_regime_id": None, "evidence_id": evidence_id,
                    "decision_id": None, "disposition": None, "limit_codes": [],
                    "history_refs": [], "explanation_id": None, "model_revision": None,
                    "prompt_sha256": None, "image_ids_sent": [], "guard_result": None,
                })
                locators[image_id] = entry.locator()
                cursors[evidence_id] = cursor
                cursor += 1
    trusted = measurement.TrustedMeasuredEvidenceResult(
        {"measurements": []}, tuple(bindings), registry,
        _factory_token=measurement._TRUSTED_MEASUREMENT_TOKEN,
    )
    provenance = {
        "schema_version": "wp7-rgb-runtime-provenance-v1",
        "vials": [
            {
                "case_id": case_id,
                "selected_frame_indices": frames,
                "measurement_citation_sha256s": measurement_citations[case_id],
            }
            for case_id, frames in selected_frames.items()
        ],
    }
    return RuntimeProjectionInput(
        tuple(templates), trusted, locators, cursors, provenance
    ), bindings


def _development_store(tmp_path: Path, bindings):
    from fluidmix.wp7_development_reference import DevelopmentReferenceStore, _create_minimal_store

    records = []
    configurations = []
    cursor = 0
    usable = [binding for binding in bindings if binding.validity == "valid"]
    first_by_tank = {
        binding.evidence_id.rsplit("-", 1)[1]: binding
        for binding in usable
        if "-CASE-1-" in binding.evidence_id
    }
    for batch, run, sign in (("INN67-DEV-RB-B01", "DONOR-A", -1),
                             ("INN67-DEV-RB-B02", "DONOR-B", 1)):
        for ordinal, tank in enumerate(("t100", "t200"), start=1):
            binding = first_by_tank[tank]
            evidence_id = f"DONOR-{batch}-{ordinal}"
            image_id = f"DONOR-IMAGE-{batch}-{ordinal}"
            rgb = tmp_path / "donors" / f"{image_id}.rgb"
            mask = tmp_path / "donors" / f"{image_id}.mask"
            rgb.parent.mkdir(exist_ok=True)
            rgb.write_bytes(evidence_id.encode())
            mask.write_bytes(image_id.encode())
            lab = list(binding.median_lab)
            lab[1] += sign * 0.5
            records.append({
                "evidence_id": evidence_id, "event_cursor": cursor, "batch_id": batch,
                "run_id": run, "case_id": f"CASE-{batch}", "recipe_id": "R-CURRENT",
                "stage_id": f"DEV-STAGE-{ordinal}", "stage_ordinal": ordinal,
                "tank_id": tank, "split_role": "development_reference",
                "image_id": image_id, "rgb_path": str(rgb), "rgb_sha256": _sha(rgb),
                "mask_path": str(mask), "mask_sha256": _sha(mask), "median_lab": lab,
                "source_render_receipt_sha256": "1" * 64,
                "source_validation_receipt_sha256": "2" * 64,
                "source_render_plan_sha256": "3" * 64,
                "known_good_profile_sha256": "4" * 64,
            })
            cursor += 1
    for ordinal, tank in enumerate(("t100", "t200"), start=1):
        configurations.append({
            "recipe_id": "R-CURRENT", "stage_id": f"DEV-STAGE-{ordinal}",
            "stage_ordinal": ordinal, "tank_id": tank,
            "peer_evidence_ids": [f"DONOR-INN67-DEV-RB-B01-{ordinal}",
                                  f"DONOR-INN67-DEV-RB-B02-{ordinal}"],
            "threshold_delta_e76": 1.5,
            "configuration_target_evidence_id": f"CONFIG-{ordinal}",
        })
    database = tmp_path / "development.sqlite3"
    _create_minimal_store(database, records=records, configurations=configurations,
        provenance={
            "development_reference_bank_sha256": "a" * 64,
            "delta_e_frozen_config_sha256": "b" * 64,
            "materialization_receipt_sha256": "c" * 64,
            "predicted_store_export_sha256": "d" * 64,
            "derived_source_manifest_sha256": "e" * 64,
            "predictor": {"producer_id": "p", "model_id": "m", "model_sha256": "8" * 64,
                          "config_sha256": "9" * 64, "code_commit": "a" * 40},
            "predictor_sha256": "f" * 64,
            "accepted_store_tuple": {"database_path": "/never/read.sqlite3",
                "database_sha256": "6" * 64, "population_receipt_path": "/never/read.json",
                "population_receipt_sha256": "7" * 64},
            "inn67_provenance": {"fixed": "test"},
        })
    digest = _sha(database)
    return database, digest, DevelopmentReferenceStore(database, expected_snapshot_sha256=digest)


def _fixture(tmp_path: Path, monkeypatch):
    import fluidmix.carryover_pipeline as pipeline
    from test_carryover_pipeline import _freeze, _recipe, _write_json

    recipe_value = _recipe(vial_count=4)
    recipe_value["runs"][0]["vials"][0]["previous_recipe_id"] = None
    recipe_value["runs"][0]["vials"][2]["recipe_id"] = "R-UNSUPPORTED"
    video = tmp_path / "video.bin"
    video.write_bytes(b"synthetic camera fixture")
    recipe_path = tmp_path / "recipe.json"
    _write_json(recipe_path, recipe_value)
    freeze, freeze_sha = _freeze(tmp_path, accepts_real_rgb_only=True)
    runtime_output, bindings = _runtime_output(tmp_path, recipe_value)
    database, snapshot, store = _development_store(tmp_path, bindings)
    capture_plan, capture_plan_sha = _write_capture_fixture(
        tmp_path, recipe_value, video
    )
    monkeypatch.setattr(pipeline, "_validate_execution_preflight", lambda *_args, **_kwargs: {
        "preflight_sha256": "e" * 64,
        "test": {"vial_count": 4}, "real_check": {"vial_count": 30},
        "ordered_video": {"ordered_video_path": str(video.resolve()),
                          "ordered_video_sha256": _sha(video)},
        "history": {"store_snapshot_sha256": snapshot},
    })

    class Runtime:
        runtime_id = "reviewed-clamp-runtime-v1"
        runtime_sha256 = "2" * 64
        config_sha256 = "3" * 64
        accepts_real_rgb_only = True

        def analyze(self, **_kwargs):
            return runtime_output

    return {
        "video": video, "recipe_sequence": recipe_path, "freeze": freeze,
        "expected_freeze_sha256": freeze_sha,
        "evidence_out_store": tmp_path / "explained.sqlite3",
        "out": tmp_path / "receipt.json", "claim": tmp_path / "claim.json",
        "owner_id": "owner", "preflight_receipt": {},
        "expected_preflight_sha256": "e" * 64,
        "selected_mode": "development_reference_backed",
        "development_reference_store": database,
        "development_reference_store_sha256": snapshot,
        "capture_plan": capture_plan,
        "capture_plan_sha256": capture_plan_sha,
        "capture_receipt_out": tmp_path / "capture.json",
        "capture_provenance_out": tmp_path / "capture-provenance.json",
    }, Runtime(), runtime_output, store


def _runtime_with_missing_and_integrity_invalid_binding(runtime_output):
    import fluidmix.vision_measurement as measurement
    from fluidmix.carryover_pipeline import RuntimeProjectionInput

    registry = measurement.trusted_prediction_registry_from_result_v1(
        runtime_output.trusted_adapter_result
    )
    bindings = list(measurement.trusted_measurement_bindings_from_result_v1(
        runtime_output.trusted_adapter_result
    ))
    missing_image = "CASE-1-F000101"
    invalid_image = "CASE-1-F000102"
    invalid_entry = registry.entry(invalid_image)
    Path(invalid_entry.rgb_path).write_bytes(b"integrity-invalid RGB")
    _measured, invalid_binding = measurement.measure_trusted_predicted_image(
        invalid_entry,
        measurement_id="MEAS-CASE-1-t200",
        evidence_id="EVID-CASE-1-t200",
    )
    assert invalid_binding.limit_codes == ("rgb_hash_mismatch",)
    retained = [
        binding for binding in bindings
        if binding.image_id not in {missing_image, invalid_image}
    ]
    retained.append(invalid_binding)
    trusted = measurement.TrustedMeasuredEvidenceResult(
        {"measurements": []}, tuple(retained), registry,
        _factory_token=measurement._TRUSTED_MEASUREMENT_TOKEN,
    )
    return RuntimeProjectionInput(
        runtime_output.record_templates,
        trusted,
        runtime_output.image_locators,
        runtime_output.event_cursors,
        runtime_output.runtime_provenance,
    )


def _runtime_with_template_identity_drift(
    runtime_output, changes: dict[str, object]
):
    from fluidmix.carryover_pipeline import RuntimeProjectionInput

    templates = deepcopy(runtime_output.record_templates)
    shift = changes.get("_shift_all_vial_ordinals")
    for template in templates:
        if isinstance(shift, int):
            template["vial_ordinal"] += shift
        if template["case_id"] == "CASE-1":
            for field, value in changes.items():
                if field.startswith("_"):
                    continue
                if field != "stage_id" or template["stage_ordinal"] == 1:
                    template[field] = value
    return RuntimeProjectionInput(
        tuple(templates),
        runtime_output.trusted_adapter_result,
        runtime_output.image_locators,
        runtime_output.event_cursors,
        runtime_output.runtime_provenance,
    )


def _runtime_with_swapped_stage_images(runtime_output):
    from fluidmix.carryover_pipeline import RuntimeProjectionInput

    templates = deepcopy(runtime_output.record_templates)
    first_vial = [
        template for template in templates if template["case_id"] == "CASE-1"
    ]
    assert len(first_vial) == 2
    first_vial[0]["image_id"], first_vial[1]["image_id"] = (
        first_vial[1]["image_id"], first_vial[0]["image_id"]
    )
    return RuntimeProjectionInput(
        tuple(templates),
        runtime_output.trusted_adapter_result,
        runtime_output.image_locators,
        runtime_output.event_cursors,
        runtime_output.runtime_provenance,
    )


def _runtime_without_nullable_identity_field(runtime_output):
    from fluidmix.carryover_pipeline import RuntimeProjectionInput

    templates = deepcopy(runtime_output.record_templates)
    for template in templates:
        if template["case_id"] == "CASE-1":
            assert template["previous_recipe_id"] is None
            template.pop("previous_recipe_id")
    return RuntimeProjectionInput(
        tuple(templates),
        runtime_output.trusted_adapter_result,
        runtime_output.image_locators,
        runtime_output.event_cursors,
        runtime_output.runtime_provenance,
    )


def _runtime_with_endpoint_unavailable_cases(runtime_output, unavailable_cases):
    import fluidmix.vision_measurement as measurement
    from fluidmix.carryover_pipeline import RuntimeProjectionInput

    registry = measurement.trusted_prediction_registry_from_result_v1(
        runtime_output.trusted_adapter_result
    )
    retained_templates = tuple(
        template for template in runtime_output.record_templates
        if template["case_id"] not in unavailable_cases
    )
    retained_image_ids = {
        template["image_id"] for template in retained_templates
    }
    retained_bindings = tuple(
        binding
        for binding in measurement.trusted_measurement_bindings_from_result_v1(
            runtime_output.trusted_adapter_result
        )
        if binding.image_id in retained_image_ids
    )
    trusted = measurement.TrustedMeasuredEvidenceResult(
        {"measurements": []}, retained_bindings, registry,
        _factory_token=measurement._TRUSTED_MEASUREMENT_TOKEN,
    )
    provenance = deepcopy(runtime_output.runtime_provenance)
    for vial in provenance["vials"]:
        if vial["case_id"] in unavailable_cases:
            vial["selection_status"] = "abstain"
            vial["limit_codes"] = ["dispense_stream_bout_missing"]
            vial["selected_frame_indices"] = []
            vial["measurement_citation_sha256s"] = []
    return RuntimeProjectionInput(
        retained_templates,
        trusted,
        {
            image_id: locator
            for image_id, locator in runtime_output.image_locators.items()
            if image_id in retained_image_ids
        },
        {
            evidence_id: cursor
            for evidence_id, cursor in runtime_output.event_cursors.items()
            if any(
                template["evidence_id"] == evidence_id
                for template in retained_templates
            )
        },
        provenance,
    )


def test_public_run_wires_keep_discard_unsupported_and_measurement_insufficient(tmp_path, monkeypatch):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    database = kwargs["development_reference_store"]
    snapshot = kwargs["development_reference_store_sha256"]
    evidence_out = kwargs["evidence_out_store"]
    claim = kwargs["claim"]
    upstream = {
        path: _sha(path)
        for path in (kwargs["video"], kwargs["recipe_sequence"], database)
    }
    projection_calls = []
    original_projection = pipeline.project_runtime_measurements

    def tracked_projection(value):
        projection_calls.append(value)
        return original_projection(value)

    monkeypatch.setattr(pipeline, "project_runtime_measurements", tracked_projection)

    def load_reference():
        assert claim.exists()
        assert projection_calls
        assert not evidence_out.exists()
        return store

    receipt = pipeline.run_test(
        **kwargs, capture_runtime_loader=_capture_loader(kwargs, runtime),
        development_reference_store_loader=load_reference,
    )

    assert [row["disposition"] for row in receipt["operator_records"]] == [
        "keep", "discard", "insufficient_evidence", "insufficient_evidence"
    ]
    assert receipt["denominators"]["decision_count"] == 2
    assert receipt["denominators"]["abstention_count"] == 2
    assert receipt["denominators"]["expected_stage_count"] == 8
    assert receipt["provenance_inventory"]["measured_stage_count"] == 6
    assert receipt["provenance_inventory"]["explained_stage_count"] == 6
    assert receipt["source_bindings"]["evidence_out_store_sha256"] == _sha(evidence_out)
    assert receipt["source_bindings"]["development_reference_store_snapshot_sha256"] == snapshot
    assert not {
        "sim_metadata", "truth", "label", "model_answer", "residue_ml",
        "tank_colour_preset",
    }.intersection(_keys(receipt))
    with sqlite3.connect(evidence_out) as connection:
        assert connection.execute("SELECT COUNT(*) FROM stages WHERE phase='explained'").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM decisions").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM explanations").fetchone() == (6,)
        assert connection.execute("SELECT COUNT(*) FROM decisions WHERE history_refs_json != '[]'").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM sim_metadata").fetchone() == (0,)
    donor_record = receipt["operator_records"][0]["source_bindings"]["development_references"][0]
    assert donor_record["donor_store_snapshot_sha256"] == snapshot
    assert len(donor_record["donors"]) == 2
    assert {path: _sha(path) for path in upstream} == upstream


def test_all_measurement_unavailable_creates_hash_bound_empty_evidence_store(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    unavailable_output = _runtime_with_endpoint_unavailable_cases(
        runtime_output, {"CASE-1", "CASE-2", "CASE-3", "CASE-4"}
    )

    class AllUnavailableRuntime:
        runtime_id = runtime.runtime_id
        runtime_sha256 = runtime.runtime_sha256
        config_sha256 = runtime.config_sha256
        accepts_real_rgb_only = True

        def analyze(self, **_values):
            return unavailable_output

    receipt = pipeline.run_test(
        **kwargs,
        capture_runtime_loader=_capture_loader(kwargs, AllUnavailableRuntime()),
        development_reference_store_loader=lambda: store,
    )

    evidence_out = kwargs["evidence_out_store"]
    assert evidence_out.is_file()
    assert receipt["terminal_status"] == "complete"
    assert receipt["denominators"]["abstention_count"] == 4
    assert receipt["denominators"]["decision_count"] == 0
    assert all(
        row["disposition"] == "insufficient_evidence"
        for row in receipt["operator_records"]
    )
    assert receipt["source_bindings"]["evidence_out_store_sha256"] == _sha(
        evidence_out
    )
    with sqlite3.connect(evidence_out) as connection:
        for table in ("stages", "decisions", "explanations", "sim_metadata"):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone() == (0,)


def test_endpoint_unavailable_vial_does_not_erase_valid_sibling_decision(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    mixed_output = _runtime_with_endpoint_unavailable_cases(
        runtime_output, {"CASE-1"}
    )

    class MixedEndpointRuntime:
        runtime_id = runtime.runtime_id
        runtime_sha256 = runtime.runtime_sha256
        config_sha256 = runtime.config_sha256
        accepts_real_rgb_only = True

        def analyze(self, **_values):
            return mixed_output

    receipt = pipeline.run_test(
        **kwargs,
        capture_runtime_loader=_capture_loader(kwargs, MixedEndpointRuntime()),
        development_reference_store_loader=lambda: store,
    )

    records = {row["case_id"]: row for row in receipt["operator_records"]}
    assert records["CASE-1"]["disposition"] == "insufficient_evidence"
    assert "measurement_unavailable" in records["CASE-1"]["reason_codes"]
    assert records["CASE-2"]["disposition"] == "discard"
    provenance = receipt["provenance_inventory"]["runtime_provenance"]
    unavailable = next(
        vial for vial in provenance["vials"] if vial["case_id"] == "CASE-1"
    )
    assert unavailable["selection_status"] == "abstain"
    assert unavailable["limit_codes"] == ["dispense_stream_bout_missing"]
    assert unavailable["measurement_citation_sha256s"] == []


@pytest.mark.parametrize("runtime_case", ["missing", "freeze_mismatch"])
def test_development_reference_runtime_identity_failure_is_fatal(
    tmp_path, monkeypatch, runtime_case
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, _runtime_output_value, store = _fixture(tmp_path, monkeypatch)
    reference_calls = []
    if runtime_case == "missing":
        selected_runtime = None
    else:
        class FreezeMismatchedRuntime:
            runtime_id = runtime.runtime_id
            runtime_sha256 = "0" * 64
            config_sha256 = runtime.config_sha256
            accepts_real_rgb_only = True

            def analyze(self, **values):
                return runtime.analyze(**values)

        selected_runtime = FreezeMismatchedRuntime()

    with pytest.raises(ValueError, match="selected reference runtime"):
        pipeline.run_test(
            **kwargs,
            capture_runtime_loader=_capture_loader(kwargs, selected_runtime),
            development_reference_store_loader=(
                lambda: reference_calls.append("reference") or store
            ),
        )

    assert reference_calls == []
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"
    assert not kwargs["evidence_out_store"].exists()


def test_missing_binding_cannot_suppress_present_integrity_failure(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    mixed_output = _runtime_with_missing_and_integrity_invalid_binding(runtime_output)

    class MixedRuntime:
        runtime_id = runtime.runtime_id
        runtime_sha256 = runtime.runtime_sha256
        config_sha256 = runtime.config_sha256
        accepts_real_rgb_only = True

        def analyze(self, **_values):
            return mixed_output

    with pytest.raises(ValueError, match="runtime measurement integrity failure"):
        pipeline.run_test(
            **kwargs,
            capture_runtime_loader=_capture_loader(kwargs, MixedRuntime()),
            development_reference_store_loader=lambda: store,
        )

    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"
    assert not kwargs["evidence_out_store"].exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"_shift_all_vial_ordinals": 10},
        {"stage_id": "STAGE-WRONG-1"},
        {"previous_recipe_id": "R-WRONG-PREVIOUS"},
    ],
)
def test_runtime_template_identity_drift_is_fatal_before_reference_load(
    tmp_path, monkeypatch, changes
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    drifted_output = _runtime_with_template_identity_drift(
        runtime_output, changes
    )
    reference_calls = []

    class DriftedRuntime:
        runtime_id = runtime.runtime_id
        runtime_sha256 = runtime.runtime_sha256
        config_sha256 = runtime.config_sha256
        accepts_real_rgb_only = True

        def analyze(self, **_values):
            return drifted_output

    with pytest.raises(ValueError, match="runtime template identity"):
        pipeline.run_test(
            **kwargs,
            capture_runtime_loader=_capture_loader(kwargs, DriftedRuntime()),
            development_reference_store_loader=(
                lambda: reference_calls.append("reference") or store
            ),
        )

    assert reference_calls == []
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"
    assert not kwargs["evidence_out_store"].exists()


@pytest.mark.parametrize(
    "mutator",
    [_runtime_with_swapped_stage_images, _runtime_without_nullable_identity_field],
    ids=["swapped_stage_images", "missing_nullable_previous_recipe"],
)
def test_runtime_stage_association_is_fatal_before_reference_load(
    tmp_path, monkeypatch, mutator
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    invalid_output = mutator(runtime_output)
    reference_calls = []

    class InvalidAssociationRuntime:
        runtime_id = runtime.runtime_id
        runtime_sha256 = runtime.runtime_sha256
        config_sha256 = runtime.config_sha256
        accepts_real_rgb_only = True

        def analyze(self, **_values):
            return invalid_output

    with pytest.raises(ValueError, match="runtime template identity"):
        pipeline.run_test(
            **kwargs,
            capture_runtime_loader=_capture_loader(
                kwargs, InvalidAssociationRuntime()
            ),
            development_reference_store_loader=(
                lambda: reference_calls.append("reference") or store
            ),
        )

    assert reference_calls == []
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"
    assert not kwargs["evidence_out_store"].exists()


def test_cli_reference_arguments_are_conditional_and_store_name_is_unambiguous():
    from scripts.run_carryover_pipeline import build_parser

    help_text = build_parser().format_help()
    run_help = build_parser()._subparsers._group_actions[0].choices["run-test"].format_help()
    assert "--evidence-out-store" in run_help
    assert "--development-reference-store" in run_help
    assert "--store" not in run_help
    assert "run-test" in help_text


@pytest.mark.parametrize("occupied_kind", ["existing", "symlink", "alias"])
def test_output_store_no_clobber_and_alias_fail_before_either_loader(
    tmp_path, monkeypatch, occupied_kind
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, _runtime_output_value, store = _fixture(tmp_path, monkeypatch)
    calls = []
    if occupied_kind == "existing":
        kwargs["evidence_out_store"].write_bytes(b"existing")
        error = FileExistsError
    elif occupied_kind == "symlink":
        kwargs["evidence_out_store"].symlink_to(tmp_path / "missing-target.sqlite3")
        error = FileExistsError
    else:
        kwargs["evidence_out_store"] = kwargs["out"]
        error = ValueError

    with pytest.raises(error):
        pipeline.run_test(
            **kwargs,
            capture_runtime_loader=(
                lambda *_args: calls.append("runtime") or runtime
            ),
            development_reference_store_loader=lambda: calls.append("reference") or store,
        )

    assert calls == []
    assert not kwargs["claim"].exists()


def test_reference_integrity_failure_is_fatal_and_upstreams_remain_immutable(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline
    from fluidmix.wp7_development_reference import DevelopmentReferenceError

    kwargs, runtime, _runtime_output_value, store = _fixture(tmp_path, monkeypatch)
    upstream = {
        path: _sha(path)
        for path in (
            kwargs["video"], kwargs["recipe_sequence"],
            kwargs["development_reference_store"],
        )
    }

    def corrupt_query(_target):
        raise DevelopmentReferenceError("configuration_donor_binding_invalid")

    monkeypatch.setattr(store, "reference_bank", corrupt_query)
    with pytest.raises(
        DevelopmentReferenceError, match="configuration_donor_binding_invalid"
    ):
        pipeline.run_test(
            **kwargs, capture_runtime_loader=_capture_loader(kwargs, runtime),
            development_reference_store_loader=lambda: store,
        )

    assert {path: _sha(path) for path in upstream} == upstream
    assert not kwargs["evidence_out_store"].exists()
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"


def test_runtime_artifact_hash_corruption_is_fatal_not_a_vial_abstention(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    kwargs, runtime, runtime_output, store = _fixture(tmp_path, monkeypatch)
    image_id = next(iter(runtime_output.image_locators))
    rgb = Path(runtime_output.image_locators[image_id]["rgb_path"])
    rgb.write_bytes(b"corrupt after trusted binding")
    upstream = {
        path: _sha(path)
        for path in (
            kwargs["video"], kwargs["recipe_sequence"],
            kwargs["development_reference_store"],
        )
    }

    with pytest.raises(ValueError, match="RGB image SHA-256 mismatch"):
        pipeline.run_test(
            **kwargs, capture_runtime_loader=_capture_loader(kwargs, runtime),
            development_reference_store_loader=lambda: store,
        )

    assert {path: _sha(path) for path in upstream} == upstream
    assert not kwargs["evidence_out_store"].exists()
    assert json.loads(kwargs["out"].read_text())["terminal_status"] == "fatal"


def test_pipeline_source_hash_covers_each_production_dependency_once(monkeypatch):
    import fluidmix.carryover_pipeline as pipeline

    observed = []
    original = pipeline._read_regular_bytes

    def tracked(path, role):
        observed.append(Path(path).relative_to(ROOT).as_posix())
        return original(path, role)

    monkeypatch.setattr(pipeline, "_read_regular_bytes", tracked)
    digest = pipeline.pipeline_source_sha256()
    assert len(digest) == 64
    assert len(observed) == len(set(observed))
    assert set(observed) == {
        "scripts/run_carryover_pipeline.py",
        "src/fluidmix/carryover_pipeline.py",
        "src/fluidmix/evidence_store.py",
        "src/fluidmix/part_b_evidence.py",
        "src/fluidmix/vision_measurement.py",
        "src/fluidmix/wp7_development_reference.py",
        "src/fluidmix/wp7_final_decision.py",
        "src/fluidmix/wp7_phase_capture.py",
        "src/fluidmix/wp7_rgb_runtime.py",
            "src/fluidmix/wp7_fill_step_production.py",
            "src/fluidmix/wp7_fill_step_tracker.py",
            "src/fluidmix/wp7_tracker_runtime.py",
            "src/fluidmix/wp7_vlm_production_materializer.py",
    }

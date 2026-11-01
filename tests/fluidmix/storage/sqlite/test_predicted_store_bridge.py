"""Bounded schema-v2 prediction to WP5 production-store bridge tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
PRODUCTION_MATERIALIZATION_RECEIPT = Path(
    "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/"
    "wp5-cnn-swap-materialization-d89a3f9-01a18128-attempt02/"
    "materialization-receipt.json"
)
PRODUCTION_MATERIALIZATION_RECEIPT_SHA256 = (
    "a9a559a3614a23c362b646d4608ee4d3f20f1a1d0599c9534946a2b79ece71e3"
)
PRODUCTION_LOCATOR_SHA256 = (
    "578ab949803214b12cd4942cff7adeda3e89610eff92523505cf7c0436e5e269"
)
PRODUCTION_MATERIALIZER_COMMIT = "d89a3f9fca31e7a2a37c89290189b3a5fa0583da"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _source(path: Path, artifact: object) -> Path:
    _write_json(path, artifact)
    return path


def _payload(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace(path: Path, value: object) -> None:
    _write_json(path, value)


def _fixture(tmp_path: Path) -> dict[str, object]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    specs = (
        ("train", "RUN-TRAIN-01", "BATCH_0000000000000001", "5", ("t200", "t400", "t100")),
        ("train", "RUN-TRAIN-02", "BATCH_0000000000000002", "6", ("t200", "t100")),
        ("train", "RUN-TRAIN-03", "BATCH_0000000000000003", "7", ("t400", "t100")),
        ("train", "RUN-TRAIN-04", "BATCH_0000000000000004", "8", ("t200", "t400", "t100")),
        ("validation", "RUN-VALIDATION-01", "BATCH_0000000000000005", "13", ("t200", "t100")),
        ("evaluation", "RUN-EVALUATION-01", "BATCH_0000000000000006", "14", ("t400", "t100")),
    )
    images: dict[str, dict[str, str]] = {}
    safe_records = []
    measurements = []
    pairs = []
    source_manifests = []
    capture_hashes: dict[str, str] = {}
    capture_paths: dict[str, Path] = {}
    source_receipt_paths: dict[tuple[str, str], Path] = {}
    publication_hashes: dict[str, str] = {}
    runtime_hashes: dict[str, str] = {}
    receipt_root = tmp_path / "accepted"
    receipt_files = []
    for spec_index, (role, run_id, batch_id, recipe_id, tanks) in enumerate(specs, 1):
        manifest = receipt_root / "publications" / batch_id / "f2-stage-capture-manifest.json"
        _write_json(manifest, {"accepted_capture": batch_id})
        capture_hashes[batch_id] = _sha(manifest)
        capture_paths[batch_id] = manifest
        source_manifests.append(
            {
                "batch_id": batch_id,
                "run_id": run_id,
                "recipe_id": recipe_id,
                "split_role": role,
                "sha256": capture_hashes[batch_id],
                "active_stage_count": len(tanks),
                "stage_measurement_count": len(tanks) * 3,
            }
        )
        source_receipts = []
        for vial in range(1, 4):
            case_id = f"CASE_{spec_index:014d}{vial:02d}"
            clip = artifact_root / "clips" / f"{case_id}.avi"
            clip.parent.mkdir(exist_ok=True)
            clip.write_bytes(f"clip:{case_id}".encode())
            source_receipt = receipt_root / "source-receipts" / f"{case_id}.json"
            _write_json(source_receipt, {"accepted_source_receipt": case_id})
            receipt_sha = _sha(source_receipt)
            source_receipt_paths[(batch_id, f"V{vial}")] = source_receipt
            source_receipts.append(
                {
                    "case_id": case_id,
                    "vial_id": f"V{vial}",
                    "vial_position": vial,
                    "receipt_sha256": receipt_sha,
                    "receipt_path": str(source_receipt.resolve()),
                }
            )
            stages = []
            by_stage = []
            for ordinal, tank in enumerate(tanks, 1):
                image_id = f"{batch_id}-V{vial}-dispense-{ordinal}-{tank}"
                rgb = artifact_root / "pixels" / f"{image_id}-rgb.png"
                mask = artifact_root / "predictions" / f"{image_id}-mask.png"
                rgb.parent.mkdir(exist_ok=True)
                mask.parent.mkdir(exist_ok=True)
                Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), mode="RGB").save(rgb)
                mask_pixels = np.zeros((3, 4), dtype=np.uint8)
                mask_pixels[1:, 1:3] = 255
                Image.fromarray(mask_pixels, mode="L").save(mask)
                images[image_id] = {
                    "rgb_path": str(rgb.resolve()),
                    "rgb_sha256": _sha(rgb),
                    "mask_path": str(mask.resolve()),
                    "mask_sha256": _sha(mask),
                }
                stage_id = f"dispense-{ordinal}-{tank}"
                evidence_id = f"EVID-{image_id}"
                stages.append(
                    {
                        "image_id": image_id,
                        "stage_id": stage_id,
                        "stage_ordinal": ordinal,
                        "tank_id": tank,
                    }
                )
                row = {
                    "evidence_id": evidence_id,
                    "measurement_id": f"MEAS-{image_id}",
                    "image_id": image_id,
                    "batch_id": batch_id,
                    "case_key": f"{batch_id}:V{vial}",
                    "run_id": run_id,
                    "recipe_id": recipe_id,
                    "split_role": role,
                    "split_version": "whole-run-varied-v3",
                    "vial_id": f"V{vial}",
                    "stage_id": stage_id,
                    "stage_ordinal": ordinal,
                    "tank_id": tank,
                    "view_id": "vial_fill",
                    "source_manifest_sha256": capture_hashes[batch_id],
                    "rgb_sha256": images[image_id]["rgb_sha256"],
                    "mask_sha256": images[image_id]["mask_sha256"],
                    "median_srgb_0_255": [0.0, 0.0, 0.0],
                    "mad_srgb_0_255": [0.0, 0.0, 0.0],
                    "median_lab": [0.0, 0.0, 0.0],
                    "mad_lab": [0.0, 0.0, 0.0],
                    "iqr_lab": [0.0, 0.0, 0.0],
                    "liquid_pixel_count": 4,
                    "mask_coverage_fraction": 1.0 / 3.0,
                    "mask_height_px": 2,
                    "lab_unit": "cie_lab_d65_2deg",
                }
                measurements.append(row)
                by_stage.append(row)
            safe_records.append(
                {
                    "batch_id": batch_id,
                    "run_id": run_id,
                    "case_id": case_id,
                    "vial_id": f"V{vial}",
                    "vial_position": vial,
                    "recipe_id": recipe_id,
                    "previous_recipe_id": str(max(0, int(recipe_id) - 1)),
                    "split_role": role,
                    "split_version": "whole-run-varied-v3",
                    "recipe_ml": {
                        "t100": 20.0 if "t100" in tanks else 0.0,
                        "t200": 20.0 if "t200" in tanks else 0.0,
                        "t400": 20.0 if "t400" in tanks else 0.0,
                    },
                    "dispense_order": list(tanks),
                    "source_receipt_sha256": receipt_sha,
                    "video_path": str(clip.resolve()),
                    "video_sha256": _sha(clip),
                    "video_decoded_frame_count": 30,
                    "video_playback_fps": 10.0,
                    "video_codec": "MJPG",
                    "video_container": "avi",
                    "fill_frame_range": [0, 20],
                    "excluded_post_fill_frame_range": [20, 30],
                    "stages": stages,
                }
            )
        for ordinal, tank in enumerate(tanks, 1):
            same_stage = [
                row
                for row in measurements
                if row["batch_id"] == batch_id and row["stage_ordinal"] == ordinal
            ]
            for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
                left, right = same_stage[left_index], same_stage[right_index]
                pairs.append(
                    {
                        "pair_id": f"PAIR-{batch_id}-{ordinal}-{left_index}-{right_index}",
                        "batch_id": batch_id,
                        "run_id": run_id,
                        "recipe_id": recipe_id,
                        "split_role": role,
                        "split_version": "whole-run-varied-v3",
                        "stage_id": f"dispense-{ordinal}-{tank}",
                        "stage_ordinal": ordinal,
                        "tank_id": tank,
                        "left_vial_id": left["vial_id"],
                        "right_vial_id": right["vial_id"],
                        "left_evidence_id": left["evidence_id"],
                        "right_evidence_id": right["evidence_id"],
                        "source_manifest_sha256": capture_hashes[batch_id],
                        "left_rgb_sha256": left["rgb_sha256"],
                        "right_rgb_sha256": right["rgb_sha256"],
                        "left_mask_sha256": left["mask_sha256"],
                        "right_mask_sha256": right["mask_sha256"],
                        "delta_e76": 0.0,
                        "delta_e76_6dp": "0.000000",
                        "unit": "delta_e76",
                    }
                )
        publication = receipt_root / "publications" / batch_id / "publication-receipt.json"
        publication_payload = {
            "schema_version": 3,
            "artifact_type": "f2_stage_batch_publication_receipt",
            "status": "published",
            "outputs": {
                "f2-stage-capture-manifest.json": {
                    "path": "f2-stage-capture-manifest.json",
                    "bytes": manifest.stat().st_size,
                    "sha256": capture_hashes[batch_id],
                }
            },
            "source_receipts": source_receipts,
        }
        _write_json(publication, publication_payload)
        publication_hashes[batch_id] = _sha(publication)
        runtime_hashes[batch_id] = hashlib.sha256(f"runtime:{batch_id}".encode()).hexdigest()
        for path in (manifest, publication):
            receipt_files.append(
                {
                    "path": path.relative_to(receipt_root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha(path),
                }
            )

    predictor_config = {"threshold": 0.5, "reference_mode": "empty_vial"}
    predictor_config_sha = hashlib.sha256(
        json.dumps(
            predictor_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    producer = {
        "producer_id": "segmenter-accepted-v1",
        "producer_kind": "rgb_only_liquid_segmenter",
        "model_id": "spatial-rf-v1",
        "model_sha256": "a" * 64,
        "config_id": "frozen-validation-v1",
        "config_sha256": predictor_config_sha,
        "code_commit": "c" * 40,
        "threshold_source": "twin_validation_frozen",
        "training_data_origin": "twin",
        "input_fields": ["rgb"],
    }
    role_by_image = {row["image_id"]: row["split_role"] for row in measurements}
    production_images = {
        image_id: locator
        for image_id, locator in images.items()
        if role_by_image[image_id] != "evaluation"
    }
    predictions = {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": producer,
        "images": production_images,
    }
    production_measurements = [
        row for row in measurements if row["split_role"] != "evaluation"
    ]
    production_pairs = [row for row in pairs if row["split_role"] != "evaluation"]
    production_sources = [
        row for row in source_manifests if row["split_role"] != "evaluation"
    ]
    measured = {
        "schema_version": 1,
        "artifact_type": "f2_varied_stage_measured_evidence",
        "status": "measured",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
        "source_manifests": production_sources,
        "stage_measurements": production_measurements,
        "same_run_stage_pairs": production_pairs,
        "measurement_contract": {"scope": "measurements_only_no_decisions"},
        "denominators": {
            "source_manifests": 5,
            "runs": 5,
            "vials": 15,
            "active_stage_events": 12,
            "stage_measurements": 36,
            "same_run_stage_pairs": 36,
            "two_stage_runs": 3,
            "three_stage_runs": 2,
        },
        "claim_boundary": "synthetic measurements only",
        "mask_source": "rgb_only_prediction",
        "mask_prediction_producer": producer,
    }
    materialization_sha = "d" * 64
    downstream = {
        "schema_version": 2,
        "artifact_type": "f2_varied_v3_downstream_validation_receipt",
        "status": "pass",
        "batch_count": 6,
        "vial_count": 18,
        "stage_pair_count": 42,
        "source_artifact_validation_count": 18,
        "capture_manifest_sha256_by_batch": capture_hashes,
        "publication_receipt_sha256_by_batch": publication_hashes,
        "runtime_receipt_sha256_by_batch": runtime_hashes,
        "materialization_receipt": {"sha256": materialization_sha},
        "files": receipt_files,
    }
    downstream_path = _source(receipt_root / "downstream-validation-receipt.json", downstream)
    safe = {
        "schema_version": 1,
        "artifact_type": "f2_fill_clip_input_manifest",
        "status": "published",
        "data_origin": "twin",
        "split_version": "whole-run-varied-v3",
        "source_downstream_validation_receipt_sha256": _sha(downstream_path),
        "source_request_materialization_receipt_sha256": materialization_sha,
        "source_image_locators_sha256": "e" * 64,
        "denominators": {"batches": 6, "runs": 6, "vials": 18, "stages": 42},
        "records": safe_records,
        "claim_boundary": "safe clip and timing projection",
    }
    prediction_path = _source(tmp_path / "predictions.json", predictions)
    safe_path = _source(tmp_path / "safe.json", safe)
    references: dict[tuple[str, str], Path] = {}
    materialization_records = []
    for row in measurements:
        image_id = row["image_id"]
        batch_vial = (row["batch_id"], row["vial_id"])
        reference = references.get(batch_vial)
        if reference is None:
            reference = artifact_root / "references" / f"{row['batch_id']}-{row['vial_id']}.png"
            reference.parent.mkdir(exist_ok=True)
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8), mode="RGB").save(reference)
            references[batch_vial] = reference
        locator = images[image_id]
        eligible = row["split_role"] != "evaluation"
        materialization_records.append(
            {
                "image_id": image_id,
                "batch_id": row["batch_id"],
                "run_id": row["run_id"],
                "vial_id": row["vial_id"],
                "split_role": row["split_role"],
                "stage_id": row["stage_id"],
                "stage_ordinal": row["stage_ordinal"],
                "rgb_path": locator["rgb_path"],
                "rgb_sha256": locator["rgb_sha256"],
                "reference_rgb_path": str(reference.resolve()),
                "reference_rgb_sha256": _sha(reference),
                "mask_path": locator["mask_path"],
                "mask_sha256": locator["mask_sha256"],
                "prediction_status": "materialized",
                "surface_row": 1,
                "visible_height_px": 2,
                "changed_pixel_count": 4,
                "source_capture_manifest_path": str(capture_paths[row["batch_id"]].resolve()),
                "source_capture_manifest_sha256": capture_hashes[row["batch_id"]],
                "source_receipt_path": str(source_receipt_paths[batch_vial].resolve()),
                "source_receipt_sha256": _sha(source_receipt_paths[batch_vial]),
                "history_eligible": eligible,
                "history_eligibility_reason": (
                    "accepted_non_evaluation_partition"
                    if eligible
                    else "evaluation_partition_excluded"
                ),
            }
        )
    accepted_recovery = tmp_path / "accepted-recovery.json"
    accepted_review = tmp_path / "accepted-review.json"
    image_locator_source = tmp_path / "source-image-locators.json"
    request_bundle = tmp_path / "request-bundle.json"
    for path, payload in (
        (accepted_recovery, {"status": "accepted"}),
        (accepted_review, {"status": "accepted"}),
        (image_locator_source, {"status": "accepted"}),
        (request_bundle, {"status": "accepted"}),
    ):
        _write_json(path, payload)
    accepted_evidence = {
        "recovered_result_path": str(accepted_recovery.resolve()),
        "recovered_result_sha256": _sha(accepted_recovery),
        "independent_review_path": str(accepted_review.resolve()),
        "independent_review_sha256": _sha(accepted_review),
    }
    source_bindings = {
        "image_locators_path": str(image_locator_source.resolve()),
        "image_locators_sha256": _sha(image_locator_source),
        "downstream_validation_receipt_path": str(downstream_path.resolve()),
        "downstream_validation_receipt_sha256": _sha(downstream_path),
        "request_bundle_path": str(request_bundle.resolve()),
        "request_bundle_sha256": _sha(request_bundle),
    }
    safe["source_image_locators_sha256"] = source_bindings["image_locators_sha256"]
    _source(safe_path, safe)
    safe_input_record_fields = {
        "image_id", "batch_id", "run_id", "vial_id", "split_role", "stage_id",
        "stage_ordinal", "rgb_path", "rgb_sha256", "reference_rgb_path",
        "reference_rgb_sha256", "source_capture_manifest_path",
        "source_capture_manifest_sha256", "source_receipt_path",
        "source_receipt_sha256",
    }
    predictor_safe_inputs = {
        "schema_version": 1,
        "artifact_type": "empty_reference_rgb_production_inputs",
        "data_origin": "twin",
        "input_fields": ["rgb"],
        "threshold_source": producer["threshold_source"],
        "split_version": "whole-run-varied-v3",
        "accepted_evidence": accepted_evidence,
        "source_bindings": source_bindings,
        "records": [
            {key: row[key] for key in safe_input_record_fields}
            for row in materialization_records
        ],
    }
    predictor_safe_path = _source(
        tmp_path / "safe-inputs.json", predictor_safe_inputs
    )
    prediction_materialization = {
        "schema_version": 1,
        "artifact_type": "empty_reference_rgb_prediction_materialization_receipt",
        "status": "materialized",
        "materializer": {},
        "predictor": {**producer, "config": predictor_config},
        "accepted_evidence": accepted_evidence,
        "safe_manifest": {
            "path": str(predictor_safe_path),
            "sha256": _sha(predictor_safe_path),
        },
        "source_bindings": source_bindings,
        "prediction_locators": {"path": str(prediction_path), "sha256": _sha(prediction_path)},
        "denominators": {
            "images": 42,
            "vials": 18,
            "runs": 6,
            "split_images": {"train": 30, "validation": 6, "evaluation": 6},
            "history_eligible_images": 36,
            "history_excluded_evaluation_images": 6,
        },
        "inference_inputs": {
            "input_fields": ["rgb"],
            "oracle_masks_opened": 0,
            "semantic_masks_opened": 0,
            "sim_metadata_inputs": 0,
            "label_inputs": 0,
            "palette_inputs": 0,
            "composition_inputs": 0,
            "recipe_ml_inputs": 0,
            "video_inputs": 0,
        },
        "recipe_invariance": {
            "recipe_fields_present_in_safe_manifest": False,
            "programmed_ml_argument_supplied": False,
            "accepted_recovery_check": "programmed_ml_stage_outputs_invariant",
        },
        "records": materialization_records,
    }
    materializer_repo = tmp_path / "materializer-source"
    tracked = (
        "src/fluidmix/perception/materializer.py",
        "cli/fluidmix/perception/materialize_empty_reference_predictions.py",
    )
    for relative in tracked:
        path = materializer_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# frozen {relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(materializer_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(materializer_repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(materializer_repo), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(materializer_repo), "add", *tracked], check=True)
    subprocess.run(
        ["git", "-C", str(materializer_repo), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    materializer_commit = subprocess.run(
        ["git", "-C", str(materializer_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    prediction_materialization["materializer"] = {
        "code_commit": materializer_commit,
        "repository_root": str(materializer_repo.resolve()),
        "tracked_files": {
            relative: {
                "path": str((materializer_repo / relative).resolve()),
                "sha256": _sha(materializer_repo / relative),
            }
            for relative in tracked
        },
        "relevant_worktree_clean": True,
    }
    return {
        "artifact_root": artifact_root,
        "predictions": prediction_path,
        "materialization": _source(
            tmp_path / "materialization-receipt.json", prediction_materialization
        ),
        "measured": _source(tmp_path / "measured.json", measured),
        "safe": safe_path,
        "downstream": downstream_path,
    }


@pytest.mark.skipif(
    not PRODUCTION_MATERIALIZATION_RECEIPT.is_file(),
    reason="accepted production prediction publication is unavailable",
)
def test_accepted_production_receipt_has_36_only_locator_and_42_accounting_rows():
    assert _sha(PRODUCTION_MATERIALIZATION_RECEIPT) == (
        PRODUCTION_MATERIALIZATION_RECEIPT_SHA256
    )
    receipt = _payload(PRODUCTION_MATERIALIZATION_RECEIPT)
    assert receipt["denominators"] == {
        "images": 42,
        "vials": 18,
        "runs": 6,
        "split_images": {"train": 30, "validation": 6, "evaluation": 6},
        "history_eligible_images": 36,
        "history_excluded_evaluation_images": 6,
    }
    assert receipt["prediction_locators"]["sha256"] == PRODUCTION_LOCATOR_SHA256
    locator_path = Path(receipt["prediction_locators"]["path"])
    assert _sha(locator_path) == PRODUCTION_LOCATOR_SHA256
    locator_ids = set(_payload(locator_path)["images"])
    eligible_ids = {
        row["image_id"] for row in receipt["records"] if row["history_eligible"]
    }
    evaluation_ids = {
        row["image_id"]
        for row in receipt["records"]
        if row["split_role"] == "evaluation"
    }
    assert len(locator_ids) == len(eligible_ids) == 36
    assert locator_ids == eligible_ids
    assert len(evaluation_ids) == 6
    assert locator_ids.isdisjoint(evaluation_ids)


def test_current_accepted_profile_is_bound_to_reviewed_receipt():
    import fluidmix.storage.sqlite.predicted_store_bridge as bridge

    if not PRODUCTION_MATERIALIZATION_RECEIPT.is_file():
        pytest.skip("accepted production prediction publication is unavailable")
    materialization = _payload(PRODUCTION_MATERIALIZATION_RECEIPT)
    bridge._validate_accepted_publication_identity(
        materialization,
        materialization_sha=PRODUCTION_MATERIALIZATION_RECEIPT_SHA256,
        prediction_sha=PRODUCTION_LOCATOR_SHA256,
    )
    assert materialization["materializer"]["code_commit"] == PRODUCTION_MATERIALIZER_COMMIT


def _run(tmp_path: Path, fixture: dict[str, object]) -> tuple[Path, dict[str, object]]:
    import fluidmix.storage.sqlite.predicted_store_bridge as bridge
    from fluidmix.storage.sqlite.predicted_store_bridge import populate_predicted_evidence_store

    output = tmp_path / "populated"
    profile = {
        "ACCEPTED_MATERIALIZATION_RECEIPT_SHA256": _sha(fixture["materialization"]),
        "ACCEPTED_PREDICTION_LOCATORS_SHA256": _sha(fixture["predictions"]),
        "ACCEPTED_MATERIALIZER_COMMIT": _payload(fixture["materialization"])[
            "materializer"
        ]["code_commit"],
    }
    original = {name: getattr(bridge, name) for name in profile}
    try:
        for name, value in profile.items():
            setattr(bridge, name, value)
        receipt = populate_predicted_evidence_store(
            prediction_locators_path=fixture["predictions"],
            materialization_receipt_path=fixture["materialization"],
            measured_evidence_path=fixture["measured"],
            fill_clip_manifest_path=fixture["safe"],
            downstream_validation_receipt_path=fixture["downstream"],
            artifact_root=fixture["artifact_root"],
            output_dir=output,
            owner_thread_id="01a18128-8dc4-7780-ace1-4d816b49d176",
        )
    finally:
        for name, value in original.items():
            setattr(bridge, name, value)
    return output, receipt


def test_populates_36_predicted_records_and_preserves_42_row_accounting(tmp_path):
    fixture = _fixture(tmp_path)

    output, receipt = _run(tmp_path, fixture)

    assert receipt["denominators"] == {
        "accounted_stage_records": 42,
        "production_stage_records": 36,
        "excluded_evaluation_stage_records": 6,
        "train_stage_records": 30,
        "validation_stage_records": 6,
        "evaluation_stage_records": 6,
        "production_vials": 15,
        "excluded_evaluation_vials": 3,
        "production_runs": 5,
        "excluded_evaluation_runs": 1,
    }
    assert len(receipt["accounting_rows"]) == 42
    assert sum(row["store_disposition"] == "stored" for row in receipt["accounting_rows"]) == 36
    assert all(
        (row["trusted_measurement_sha256"] is None)
        == (row["split_role"] == "evaluation")
        for row in receipt["accounting_rows"]
    )
    with sqlite3.connect(output / "evidence.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM stages").fetchone() == (36,)
        assert connection.execute("SELECT DISTINCT mask_source FROM images").fetchall() == [("predicted",)]
        assert connection.execute("SELECT count(*) FROM decisions").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM explanations").fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM history_members").fetchone() == (0,)
    assert json.loads((output / "population-receipt.json").read_text()) == receipt


def test_cnn_swap_accepts_new_selection_evidence_but_preserves_safe_input_lineage(
    tmp_path, monkeypatch
):
    import fluidmix.storage.sqlite.predicted_store_bridge as bridge

    fixture = _fixture(tmp_path)
    replay = tmp_path / "cnn-replay.json"
    review = tmp_path / "cnn-review.md"
    replay.write_text('{"status":"swap_review_ready"}\n', encoding="utf-8")
    review.write_text("PASS\n", encoding="utf-8")
    config = {
        "architecture": "six-channel-resnet18-unet",
        "checkpoint_sha256": "3" * 64,
        "probability_threshold": 0.5,
    }
    producer = {
        "producer_id": bridge.CNN_PRODUCER_ID,
        "producer_kind": "rgb_only_liquid_segmenter",
        "model_id": "resnet18-unet-current-empty-rgb",
        "model_sha256": "3" * 64,
        "config_id": "wp5-cnn-six-channel-frozen-v1",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "code_commit": "a5f29ea90272a3424d8c7de09dbf435509d28105",
        "threshold_source": "twin_validation_frozen",
        "training_data_origin": "twin",
        "input_fields": ["rgb"],
    }
    predictions = _payload(fixture["predictions"])
    predictions["producer"] = producer
    _replace(fixture["predictions"], predictions)
    measured = _payload(fixture["measured"])
    measured["mask_prediction_producer"] = producer
    _replace(fixture["measured"], measured)
    materialization = _payload(fixture["materialization"])
    materialization["predictor"] = {**producer, "config": config}
    materialization["accepted_evidence"] = {
        "recovered_result_path": str(replay.resolve()),
        "recovered_result_sha256": _sha(replay),
        "independent_review_path": str(review.resolve()),
        "independent_review_sha256": _sha(review),
    }
    materialization["prediction_locators"] = {
        "path": str(Path(fixture["predictions"]).resolve()),
        "sha256": _sha(fixture["predictions"]),
    }
    materialization["recipe_invariance"]["accepted_recovery_check"] = (
        "cnn_model_inputs_exclude_recipe_ml"
    )
    _replace(fixture["materialization"], materialization)
    monkeypatch.setattr(bridge, "CNN_REPLAY_SHA256", _sha(replay))
    monkeypatch.setattr(bridge, "CNN_REVIEW_SHA256", _sha(review))

    output, receipt = _run(tmp_path, fixture)

    assert output.is_dir()
    assert receipt["prediction_producer"] == producer
    assert receipt["denominators"]["production_stage_records"] == 36


def test_self_consistent_nonaccepted_publication_fails_closed(tmp_path):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError, populate_predicted_evidence_store

    fixture = _fixture(tmp_path)
    with pytest.raises(PredictedStoreBridgeError, match="accepted_publication_identity_mismatch"):
        populate_predicted_evidence_store(
            prediction_locators_path=fixture["predictions"],
            materialization_receipt_path=fixture["materialization"],
            measured_evidence_path=fixture["measured"],
            fill_clip_manifest_path=fixture["safe"],
            downstream_validation_receipt_path=fixture["downstream"],
            artifact_root=fixture["artifact_root"],
            output_dir=tmp_path / "unaccepted",
            owner_thread_id="01a18128-8dc4-7780-ace1-4d816b49d176",
        )


def test_fill_clip_locator_lineage_mismatch_fails_before_publication(tmp_path):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError

    fixture = _fixture(tmp_path)
    safe = _payload(fixture["safe"])
    safe["source_image_locators_sha256"] = "0" * 64
    _replace(fixture["safe"], safe)

    with pytest.raises(PredictedStoreBridgeError, match="source_image_locators_lineage_mismatch"):
        _run(tmp_path, fixture)
    assert not (tmp_path / "populated").exists()


def test_evaluation_is_excluded_and_history_is_strictly_earlier(tmp_path):
    from fluidmix.storage.sqlite.evidence_store import EvidenceStore

    fixture = _fixture(tmp_path)
    output, receipt = _run(tmp_path, fixture)
    evaluation_rows = [
        row for row in receipt["accounting_rows"] if row["split_role"] == "evaluation"
    ]
    assert len(evaluation_rows) == 6
    assert {row["store_disposition"] for row in evaluation_rows} == {
        "excluded_evaluation"
    }
    assert [row["event_cursor"] for row in evaluation_rows] == list(range(36, 42))

    with sqlite3.connect(output / "evidence.sqlite3") as connection:
        assert connection.execute(
            "SELECT count(*) FROM stages WHERE split_role = 'evaluation'"
        ).fetchone() == (0,)
        assert connection.execute("SELECT count(*) FROM regime_references").fetchone() == (0,)
        serialized_pairs = " ".join(
            row[0]
            for row in connection.execute(
                "SELECT pairwise_delta_e76_json FROM measurements"
            )
        )
        assert "RUN-EVALUATION" not in serialized_pairs
        target_id, before, run_id = connection.execute(
            "SELECT evidence_id, event_cursor, run_id FROM stages "
            "ORDER BY event_cursor DESC LIMIT 1"
        ).fetchone()

    store = EvidenceStore(output / "evidence.sqlite3", artifact_root=fixture["artifact_root"])
    query = store.vial_history(
        run_id,
        before=before,
        allowed_roles={"train", "validation"},
        target_evidence_id=target_id,
    )
    assert query.members
    assert all(member.event_cursor < before for member in query.members)
    assert all(member.split_role in {"train", "validation"} for member in query.members)


def test_evaluation_masks_never_enter_registry_measurement(tmp_path, monkeypatch):
    import fluidmix.storage.sqlite.predicted_store_bridge as bridge

    fixture = _fixture(tmp_path)
    evaluation_ids = {
        row["image_id"]
        for row in _payload(fixture["materialization"])["records"]
        if row["split_role"] == "evaluation"
    }
    measured_ids = []
    original = bridge.measure_trusted_predicted_image

    def observed(entry, *, measurement_id, evidence_id):
        measured_ids.append(entry.image_id)
        return original(
            entry, measurement_id=measurement_id, evidence_id=evidence_id
        )

    monkeypatch.setattr(bridge, "measure_trusted_predicted_image", observed)

    _run(tmp_path, fixture)

    assert len(measured_ids) == 36
    assert evaluation_ids.isdisjoint(measured_ids)


@pytest.mark.parametrize("failure", ["locator_hash", "evaluation_eligible"])
def test_materialization_receipt_cannot_broaden_production_registry(tmp_path, failure):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError

    fixture = _fixture(tmp_path)
    materialization = _payload(fixture["materialization"])
    if failure == "locator_hash":
        materialization["prediction_locators"]["sha256"] = "f" * 64
    else:
        evaluation = next(
            row
            for row in materialization["records"]
            if row["split_role"] == "evaluation"
        )
        evaluation["history_eligible"] = True
    _replace(fixture["materialization"], materialization)

    with pytest.raises(PredictedStoreBridgeError):
        _run(tmp_path, fixture)


@pytest.mark.parametrize(
    ("source", "field", "value"),
    [
        ("measured", "semantic_mask_path", "/forbidden/semantic.png"),
        ("measured", "oracle", True),
        ("safe", "label", "discard"),
        ("safe", "sim_metadata", {}),
    ],
)
def test_oracle_and_semantic_fields_fail_closed(tmp_path, source, field, value):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError

    fixture = _fixture(tmp_path)
    artifact = _payload(fixture[source])
    collection = "stage_measurements" if source == "measured" else "records"
    artifact[collection][0][field] = value
    _replace(fixture[source], artifact)

    with pytest.raises(PredictedStoreBridgeError):
        _run(tmp_path, fixture)


def test_oracle_mask_path_cannot_be_relabelled_as_prediction(tmp_path):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError

    fixture = _fixture(tmp_path)
    artifact = _payload(fixture["predictions"])
    image_id, locator = next(iter(artifact["images"].items()))
    oracle = fixture["artifact_root"] / "masks" / "oracle" / f"{image_id}.png"
    oracle.parent.mkdir(parents=True)
    oracle.write_bytes(Path(locator["mask_path"]).read_bytes())
    locator["mask_path"] = str(oracle.resolve())
    locator["mask_sha256"] = _sha(oracle)
    _replace(fixture["predictions"], artifact)

    with pytest.raises(PredictedStoreBridgeError, match="trusted_prediction_registry_invalid"):
        _run(tmp_path, fixture)


@pytest.mark.parametrize("failure", ["wrong_hash", "outside_root"])
def test_wrong_prediction_hash_or_path_fails_before_store_publication(tmp_path, failure):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError

    fixture = _fixture(tmp_path)
    artifact = _payload(fixture["predictions"])
    image_id, locator = next(iter(artifact["images"].items()))
    if failure == "wrong_hash":
        locator["rgb_sha256"] = "f" * 64
    else:
        outside = tmp_path / "outside-rgb.png"
        outside.write_bytes(Path(locator["rgb_path"]).read_bytes())
        locator["rgb_path"] = str(outside.resolve())
        locator["rgb_sha256"] = _sha(outside)
    _replace(fixture["predictions"], artifact)
    materialization = _payload(fixture["materialization"])
    materialization["prediction_locators"]["sha256"] = _sha(fixture["predictions"])
    accounting = next(
        row for row in materialization["records"] if row["image_id"] == image_id
    )
    accounting["rgb_path"] = locator["rgb_path"]
    accounting["rgb_sha256"] = locator["rgb_sha256"]
    _replace(fixture["materialization"], materialization)

    with pytest.raises(PredictedStoreBridgeError):
        _run(tmp_path, fixture)
    assert not (tmp_path / "populated").exists()


@pytest.mark.parametrize("failure", ["vacuous", "duplicate_measurement", "duplicate_pair", "test_role"])
def test_vacuous_duplicate_and_test_role_inputs_fail_closed(tmp_path, failure):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError

    fixture = _fixture(tmp_path)
    if failure == "test_role":
        artifact = _payload(fixture["safe"])
        artifact["records"][0]["split_role"] = "test"
        _replace(fixture["safe"], artifact)
    else:
        artifact = _payload(fixture["measured"])
        if failure == "vacuous":
            artifact["stage_measurements"] = []
        elif failure == "duplicate_measurement":
            artifact["stage_measurements"][-1] = deepcopy(
                artifact["stage_measurements"][0]
            )
        else:
            artifact["same_run_stage_pairs"][-1] = deepcopy(
                artifact["same_run_stage_pairs"][0]
            )
        _replace(fixture["measured"], artifact)

    with pytest.raises(PredictedStoreBridgeError):
        _run(tmp_path, fixture)


def test_source_row_order_does_not_change_deterministic_event_cursors(tmp_path):
    fixture = _fixture(tmp_path)
    measured = _payload(fixture["measured"])
    measured["source_manifests"].reverse()
    measured["stage_measurements"].reverse()
    measured["same_run_stage_pairs"].reverse()
    _replace(fixture["measured"], measured)
    safe = _payload(fixture["safe"])
    safe["records"].reverse()
    _replace(fixture["safe"], safe)

    output, receipt = _run(tmp_path, fixture)

    assert [row["event_cursor"] for row in receipt["accounting_rows"]] == list(range(42))
    assert [row["split_role"] for row in receipt["accounting_rows"][:30]] == [
        "train"
    ] * 30
    with sqlite3.connect(output / "evidence.sqlite3") as connection:
        rows = connection.execute(
            "SELECT run_id, event_cursor FROM stages ORDER BY event_cursor"
        ).fetchall()
    assert [cursor for _, cursor in rows] == list(range(36))


def test_fresh_output_is_no_clobber(tmp_path):
    from fluidmix.storage.sqlite.predicted_store_bridge import PredictedStoreBridgeError

    fixture = _fixture(tmp_path)
    _run(tmp_path, fixture)

    with pytest.raises(PredictedStoreBridgeError, match="output_dir_must_be_fresh"):
        _run(tmp_path, fixture)


def test_cli_rejects_nonaccepted_publication(tmp_path):
    fixture = _fixture(tmp_path)
    output = tmp_path / "cli-populated"
    completed = subprocess.run(
        [
            sys.executable,
            "cli/fluidmix/storage/sqlite/populate_predicted_evidence_store.py",
            "--prediction-locators",
            str(fixture["predictions"]),
            "--materialization-receipt",
            str(fixture["materialization"]),
            "--measured-evidence",
            str(fixture["measured"]),
            "--fill-clip-manifest",
            str(fixture["safe"]),
            "--downstream-validation-receipt",
            str(fixture["downstream"]),
            "--artifact-root",
            str(fixture["artifact_root"]),
            "--output-dir",
            str(output),
            "--owner-thread-id",
            "01a18128-8dc4-7780-ace1-4d816b49d176",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "accepted_publication_identity_mismatch" in completed.stderr
    assert not output.exists()

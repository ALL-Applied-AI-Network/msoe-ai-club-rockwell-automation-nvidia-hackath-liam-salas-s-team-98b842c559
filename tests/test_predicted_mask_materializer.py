"""Production empty-reference predicted-mask materializer contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _synthetic_git_revision(monkeypatch):
    from fluidmix import predicted_mask_materializer as materializer

    real_verifier = materializer._verify_runtime_revision

    def verified(claim: str) -> dict[str, object]:
        return {
            "code_commit": claim,
            "repository_root": str(ROOT),
            "tracked_files": {},
            "relevant_worktree_clean": True,
        }

    monkeypatch.setattr(materializer, "_verify_runtime_revision", verified)
    return real_verifier


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rgb(path: Path, value: np.ndarray) -> None:
    Image.fromarray(value, mode="RGB").save(path)


def _safe_manifest(tmp_path: Path) -> Path:
    from fluidmix.empty_reference_rgb_predictor import fillcam_masks

    tmp_path.mkdir(parents=True, exist_ok=True)
    reference = np.full((480, 640, 3), 100, dtype=np.uint8)
    rgb = reference.copy()
    roi, _ = fillcam_masks()
    rows = np.zeros(roi.shape, dtype=bool)
    rows[271:405] = True
    rgb[roi & rows] = 180
    reference_path = tmp_path / "same-run-prepour.png"
    rgb_path = tmp_path / "stage.png"
    _write_rgb(reference_path, reference)
    _write_rgb(rgb_path, rgb)
    accepted_result = tmp_path / "accepted-result.json"
    accepted_review = tmp_path / "accepted-review.md"
    accepted_result.write_text('{"status":"evaluation_gate_pass"}\n', encoding="utf-8")
    accepted_review.write_text("PASS\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "artifact_type": "empty_reference_rgb_production_inputs",
        "data_origin": "twin",
        "input_fields": ["rgb"],
        "threshold_source": "twin_validation_frozen",
        "split_version": "whole-run-varied-v3",
        "accepted_evidence": {
            "recovered_result_path": str(accepted_result.resolve()),
            "recovered_result_sha256": _sha(accepted_result),
            "independent_review_path": str(accepted_review.resolve()),
            "independent_review_sha256": _sha(accepted_review),
        },
        "source_bindings": {
            "image_locators_path": str((tmp_path / "source-locators.json").resolve()),
            "image_locators_sha256": "a" * 64,
            "downstream_validation_receipt_path": str((tmp_path / "validation.json").resolve()),
            "downstream_validation_receipt_sha256": "b" * 64,
            "request_bundle_path": str((tmp_path / "requests.json").resolve()),
            "request_bundle_sha256": "c" * 64,
        },
        "records": [
            {
                "image_id": "BATCH-1-V1-dispense-1-t200",
                "batch_id": "BATCH-1",
                "run_id": "RUN-1",
                "vial_id": "V1",
                "split_role": "validation",
                "stage_id": "dispense-1-t200",
                "stage_ordinal": 1,
                "rgb_path": str(rgb_path.resolve()),
                "rgb_sha256": _sha(rgb_path),
                "reference_rgb_path": str(reference_path.resolve()),
                "reference_rgb_sha256": _sha(reference_path),
                "source_capture_manifest_path": str((tmp_path / "capture.json").resolve()),
                "source_capture_manifest_sha256": "d" * 64,
                "source_receipt_path": str((tmp_path / "receipt.json").resolve()),
                "source_receipt_sha256": "e" * 64,
            }
        ],
    }
    path = tmp_path / "safe-inputs.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _safe_manifest_for_roles(tmp_path: Path, roles: list[str]) -> Path:
    path = _safe_manifest(tmp_path)
    artifact = json.loads(path.read_bytes())
    template = artifact["records"][0]
    reference = np.asarray(Image.open(template["reference_rgb_path"]), dtype=np.uint8)
    rgb = np.asarray(Image.open(template["rgb_path"]), dtype=np.uint8)
    records = []
    for index, role in enumerate(roles, start=1):
        reference_path = tmp_path / f"same-run-prepour-{index:02}.png"
        rgb_path = tmp_path / f"stage-{index:02}.png"
        _write_rgb(reference_path, reference)
        _write_rgb(rgb_path, rgb)
        record = dict(template)
        record.update(
            {
                "image_id": f"BATCH-{index:02}-V1-dispense-1-t200",
                "batch_id": f"BATCH-{index:02}",
                "run_id": f"RUN-{index:02}",
                "split_role": role,
                "rgb_path": str(rgb_path.resolve()),
                "rgb_sha256": _sha(rgb_path),
                "reference_rgb_path": str(reference_path.resolve()),
                "reference_rgb_sha256": _sha(reference_path),
                "source_receipt_path": str(
                    (tmp_path / f"receipt-{index:02}.json").resolve()
                ),
            }
        )
        records.append(record)
    artifact["records"] = records
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_materializes_schema_v2_locator_mask_and_receipt_from_rgb_only(tmp_path):
    from fluidmix.predicted_mask_materializer import materialize_predictions
    from fluidmix.rgb_mask_prediction import validate_rgb_mask_prediction_locators

    safe_manifest = _safe_manifest(tmp_path)
    output = tmp_path / "published"

    result = materialize_predictions(
        safe_manifest_path=safe_manifest,
        output_dir=output,
        materializer_commit="f" * 40,
        require_accepted_v3=False,
    )

    locator = json.loads((output / "prediction-locators.json").read_bytes())
    images, producer = validate_rgb_mask_prediction_locators(locator)
    assert set(images) == {"BATCH-1-V1-dispense-1-t200"}
    assert producer["input_fields"] == ["rgb"]
    assert producer["threshold_source"] == "twin_validation_frozen"
    mask = Image.open(images["BATCH-1-V1-dispense-1-t200"]["mask_path"])
    assert mask.mode == "L" and mask.size == (640, 480)
    assert np.count_nonzero(np.asarray(mask)) > 0
    receipt = json.loads((output / "materialization-receipt.json").read_bytes())
    assert receipt["denominators"]["images"] == 1
    assert receipt["inference_inputs"] == {
        "input_fields": ["rgb"],
        "oracle_masks_opened": 0,
        "semantic_masks_opened": 0,
        "sim_metadata_inputs": 0,
        "label_inputs": 0,
        "palette_inputs": 0,
        "composition_inputs": 0,
        "recipe_ml_inputs": 0,
        "video_inputs": 0,
    }
    assert result["status"] == "materialized"


def test_materializes_frozen_cnn_profile_without_changing_partition_contract(
    tmp_path, monkeypatch
):
    from fluidmix import predicted_mask_materializer as materializer
    from fluidmix.empty_reference_rgb_predictor import fillcam_masks

    safe_manifest = _safe_manifest(tmp_path)
    output = tmp_path / "cnn-published"
    roi, clamp = fillcam_masks()
    mask = np.zeros((480, 640), dtype=bool)
    mask[roi & ~clamp] = True
    config = {
        "architecture": "six-channel-resnet18-unet",
        "checkpoint_sha256": "3" * 64,
        "probability_threshold": 0.5,
    }
    producer = {
        "producer_id": "frozen-wp5-six-channel-cnn",
        "producer_kind": "rgb_only_liquid_segmenter",
        "model_id": "resnet18-unet-current-empty-rgb",
        "model_sha256": "3" * 64,
        "config_id": "wp5-cnn-six-channel-frozen-v1",
        "config_sha256": materializer._canonical_sha256(config),
        "code_commit": "a5f29ea90272a3424d8c7de09dbf435509d28105",
        "threshold_source": "twin_validation_frozen",
        "training_data_origin": "twin",
        "input_fields": ["rgb"],
    }
    replay = tmp_path / "replay.json"
    review = tmp_path / "review.md"
    replay.write_text('{"status":"swap_review_ready"}\n', encoding="utf-8")
    review.write_text("PASS\n", encoding="utf-8")
    accepted = {
        "recovered_result_path": str(replay.resolve()),
        "recovered_result_sha256": _sha(replay),
        "independent_review_path": str(review.resolve()),
        "independent_review_sha256": _sha(review),
    }
    monkeypatch.setattr(
        materializer,
        "_load_frozen_cnn",
        lambda **_: (object(), producer, config, accepted),
    )
    monkeypatch.setattr(
        materializer,
        "_predict_cnn_pair",
        lambda _model, _rgb, _reference: SimpleNamespace(
            mask=mask,
            status="visible_measured",
            surface_row=0,
            visible_height_px=480,
            changed_pixel_count=int(np.count_nonzero(mask)),
        ),
    )

    result = materializer.materialize_predictions(
        safe_manifest_path=safe_manifest,
        output_dir=output,
        materializer_commit="f" * 40,
        require_accepted_v3=False,
        cnn_checkpoint_path=tmp_path / "checkpoint.pt",
        cnn_replay_path=tmp_path / "replay.json",
        cnn_review_path=tmp_path / "review.md",
    )

    locator = json.loads((output / "prediction-locators.json").read_bytes())
    assert locator["producer"] == producer
    assert result["predictor"] == {**producer, "config": config}
    assert result["accepted_evidence"] == accepted
    assert result["denominators"]["history_eligible_images"] == 1
    assert result["inference_inputs"]["oracle_masks_opened"] == 0
    assert result["recipe_invariance"] == {
        "recipe_fields_present_in_safe_manifest": False,
        "programmed_ml_argument_supplied": False,
        "accepted_recovery_check": "cnn_model_inputs_exclude_recipe_ml",
    }


def test_frozen_cnn_materialization_requires_complete_binding_set(tmp_path):
    from fluidmix.predicted_mask_materializer import (
        PredictedMaskMaterializationError,
        materialize_predictions,
    )

    output = tmp_path / "must-remain-absent"
    with pytest.raises(
        PredictedMaskMaterializationError, match="CNN binding set is incomplete"
    ):
        materialize_predictions(
            safe_manifest_path=_safe_manifest(tmp_path),
            output_dir=output,
            materializer_commit="f" * 40,
            require_accepted_v3=False,
            cnn_checkpoint_path=tmp_path / "checkpoint.pt",
        )
    assert not output.exists()


def test_absolute_path_validator_accepts_path_objects(tmp_path):
    from fluidmix.predicted_mask_materializer import _absolute_path

    source = (tmp_path / "checkpoint.pt").resolve()
    assert _absolute_path(source, "checkpoint") == source


def test_canonical_locator_registry_contains_exactly_36_non_evaluation_rows(tmp_path):
    from fluidmix.predicted_mask_materializer import materialize_predictions
    from fluidmix.rgb_mask_prediction import build_trusted_predicted_mask_registry_v1

    roles = ["train"] * 30 + ["validation"] * 6 + ["evaluation"] * 6
    safe = _safe_manifest_for_roles(tmp_path, roles)
    output = tmp_path / "published"

    receipt = materialize_predictions(
        safe_manifest_path=safe,
        output_dir=output,
        materializer_commit="f" * 40,
        require_accepted_v3=False,
    )

    locator = json.loads((output / "prediction-locators.json").read_bytes())
    registry = build_trusted_predicted_mask_registry_v1(locator)
    evaluation_ids = {
        record["image_id"]
        for record in receipt["records"]
        if record["split_role"] == "evaluation"
    }
    assert len(registry.image_ids) == 36
    assert evaluation_ids
    assert evaluation_ids.isdisjoint(registry.image_ids)
    assert all(
        Path(record["mask_path"]).is_file()
        for record in receipt["records"]
        if record["image_id"] in evaluation_ids
    )
    assert receipt["denominators"]["images"] == 42
    assert len(receipt["records"]) == 42


@pytest.mark.parametrize("drift_field", ["rgb_path", "reference_rgb_path"])
def test_late_row_rgb_or_reference_drift_leaves_output_absent(tmp_path, drift_field):
    from fluidmix.predicted_mask_materializer import (
        PredictedMaskMaterializationError,
        materialize_predictions,
    )

    safe = _safe_manifest_for_roles(tmp_path, ["validation", "validation"])
    artifact = json.loads(safe.read_bytes())
    drifted = Path(artifact["records"][1][drift_field])
    _write_rgb(drifted, np.zeros((480, 640, 3), dtype=np.uint8))
    output = tmp_path / "must-remain-absent"

    with pytest.raises(PredictedMaskMaterializationError, match="SHA-256 mismatch"):
        materialize_predictions(
            safe_manifest_path=safe,
            output_dir=output,
            materializer_commit="f" * 40,
            require_accepted_v3=False,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    "false_claim",
    ["0" * 40, "27c6699bae735c402a978687e34287520f61fedb"],
)
def test_materializer_rejects_zero_or_stale_commit_claim_before_output(
    tmp_path, false_claim, _synthetic_git_revision, monkeypatch
):
    from fluidmix import predicted_mask_materializer as materializer

    safe = _safe_manifest(tmp_path)
    output = tmp_path / "must-remain-absent"
    monkeypatch.setattr(
        materializer, "_verify_runtime_revision", _synthetic_git_revision
    )
    with pytest.raises(
        materializer.PredictedMaskMaterializationError, match="Git HEAD"
    ):
        materializer.materialize_predictions(
            safe_manifest_path=safe,
            output_dir=output,
            materializer_commit=false_claim,
            require_accepted_v3=False,
        )
    assert not output.exists()


def test_git_revision_attestation_binds_clean_head_module_and_launcher(
    tmp_path, _synthetic_git_revision, monkeypatch
):
    from fluidmix import predicted_mask_materializer as materializer

    repository = tmp_path / "repository"
    module_path = repository / "src/fluidmix/predicted_mask_materializer.py"
    launcher_path = repository / "scripts/materialize_empty_reference_predictions.py"
    module_path.parent.mkdir(parents=True)
    launcher_path.parent.mkdir(parents=True)
    module_path.write_text("MODULE = 'tracked'\n", encoding="utf-8")
    launcher_path.write_text("LAUNCHER = 'tracked'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"], check=True
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True
    )
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(materializer, "_REPOSITORY_ROOT", repository.resolve())

    provenance = _synthetic_git_revision(head)

    assert provenance["code_commit"] == head
    assert provenance["relevant_worktree_clean"] is True
    assert set(provenance["tracked_files"]) == {
        "src/fluidmix/predicted_mask_materializer.py",
        "scripts/materialize_empty_reference_predictions.py",
    }
    module_path.write_text("MODULE = 'dirty'\n", encoding="utf-8")
    with pytest.raises(
        materializer.PredictedMaskMaterializationError, match="dirty relevant files"
    ):
        _synthetic_git_revision(head)


@pytest.mark.skip(reason="requires preserved heldout-bound evidence; not authorized here")
def test_prepares_exact_42_corrected_v3_rgb_reference_bindings(tmp_path):
    from fluidmix.predicted_mask_materializer import prepare_v3_safe_manifest

    repo_root = Path("/home/hackathon5/liamreporoot")
    locators = repo_root / "f2-varied-v3-measurements-20261027/image-locators.json"
    if not locators.is_file():
        pytest.skip("preserved corrected V3 artifacts are unavailable")
    output = tmp_path / "safe-inputs.json"

    artifact = prepare_v3_safe_manifest(
        image_locators_path=locators,
        downstream_validation_receipt_path=(
            repo_root
            / "f2-varied-v3-downstream-v2-20261027/downstream-validation-receipt.json"
        ),
        request_bundle_path=(
            repo_root
            / "f2-varied-v3-downstream-v2-20261027/requests/request-materialization-receipt.json"
        ),
        recovered_result_path=(
            repo_root
            / "outputs/pipeline-20261028/rgb-mask-predictor-01a18128/"
            "approach-a-heldout-onepass-aa46da6-01a18128/recovered-publication/result.json"
        ),
        independent_review_path=(
            repo_root
            / "outputs/pipeline-20261028/rgb-mask-predictor-01a18128/reviews/"
            "predictor-publish-recovery-27c6699-independent-review.md"
        ),
        output_path=output,
    )

    assert len(artifact["records"]) == 42
    assert {record["split_role"] for record in artifact["records"]} == {
        "train",
        "validation",
        "evaluation",
    }
    assert len({record["image_id"] for record in artifact["records"]}) == 42
    assert all(record["rgb_path"] != record["reference_rgb_path"] for record in artifact["records"])
    encoded = output.read_text(encoding="utf-8").lower()
    for forbidden in ("recipe_ml", "sim_metadata", "label", "palette", "composition"):
        assert forbidden not in encoded


def test_rejects_extra_oracle_field_before_image_reads_and_never_clobbers(tmp_path, monkeypatch):
    from fluidmix.predicted_mask_materializer import (
        PredictedMaskMaterializationError,
        materialize_predictions,
    )

    safe = _safe_manifest(tmp_path)
    artifact = json.loads(safe.read_bytes())
    artifact["records"][0]["oracle_mask_path"] = "/must/not/open.png"
    safe.write_text(json.dumps(artifact), encoding="utf-8")
    opened_png = False
    original_open = Image.open

    def counted_open(*args, **kwargs):
        nonlocal opened_png
        opened_png = True
        return original_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", counted_open)
    with pytest.raises(PredictedMaskMaterializationError, match="record fields"):
        materialize_predictions(
            safe_manifest_path=safe,
            output_dir=tmp_path / "rejected",
            materializer_commit="f" * 40,
            require_accepted_v3=False,
        )
    assert opened_png is False

    clean = _safe_manifest(tmp_path / "clean")
    destination = tmp_path / "already-exists"
    destination.mkdir()
    sentinel = destination / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        materialize_predictions(
            safe_manifest_path=clean,
            output_dir=destination,
            materializer_commit="f" * 40,
            require_accepted_v3=False,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.skip(reason="requires preserved heldout-bound evidence; not authorized here")
def test_exact_prepared_v3_materializes_and_excludes_evaluation_from_history(tmp_path):
    from fluidmix.predicted_mask_materializer import (
        ACCEPTED_V3_SAFE_MANIFEST_SHA256,
        materialize_predictions,
        prepare_v3_safe_manifest,
    )

    repo_root = Path("/home/hackathon5/liamreporoot")
    if not (repo_root / "f2-varied-v3-measurements-20261027/image-locators.json").is_file():
        pytest.skip("preserved corrected V3 artifacts are unavailable")
    safe = tmp_path / "safe-inputs.json"
    prepare_v3_safe_manifest(
        image_locators_path=repo_root / "f2-varied-v3-measurements-20261027/image-locators.json",
        downstream_validation_receipt_path=repo_root / "f2-varied-v3-downstream-v2-20261027/downstream-validation-receipt.json",
        request_bundle_path=repo_root / "f2-varied-v3-downstream-v2-20261027/requests/request-materialization-receipt.json",
        recovered_result_path=repo_root / "outputs/pipeline-20261028/rgb-mask-predictor-01a18128/approach-a-heldout-onepass-aa46da6-01a18128/recovered-publication/result.json",
        independent_review_path=repo_root / "outputs/pipeline-20261028/rgb-mask-predictor-01a18128/reviews/predictor-publish-recovery-27c6699-independent-review.md",
        output_path=safe,
    )
    assert _sha(safe) == ACCEPTED_V3_SAFE_MANIFEST_SHA256

    receipt = materialize_predictions(
        safe_manifest_path=safe,
        output_dir=tmp_path / "published-v3",
        materializer_commit="f" * 40,
    )

    assert receipt["denominators"] == {
        "images": 42,
        "vials": 18,
        "runs": 6,
        "split_images": {"evaluation": 6, "train": 30, "validation": 6},
        "history_eligible_images": 36,
        "history_excluded_evaluation_images": 6,
    }
    evaluation = [row for row in receipt["records"] if row["split_role"] == "evaluation"]
    assert len(evaluation) == 6
    assert all(row["history_eligible"] is False for row in evaluation)
    assert {row["history_eligibility_reason"] for row in evaluation} == {
        "evaluation_partition_excluded"
    }


def test_production_mode_rejects_nonaccepted_safe_manifest_before_output(tmp_path):
    from fluidmix.predicted_mask_materializer import (
        PredictedMaskMaterializationError,
        materialize_predictions,
    )

    safe = _safe_manifest(tmp_path)
    output = tmp_path / "must-not-exist"
    with pytest.raises(PredictedMaskMaterializationError, match="exact accepted V3"):
        materialize_predictions(
            safe_manifest_path=safe,
            output_dir=output,
            materializer_commit="f" * 40,
        )
    assert not output.exists()


@pytest.mark.skip(reason="requires preserved heldout-bound evidence; not authorized here")
def test_production_mode_rechecks_record_provenance_before_output(tmp_path, monkeypatch):
    from fluidmix import predicted_mask_materializer as materializer

    repo_root = Path("/home/hackathon5/liamreporoot")
    if not (repo_root / "f2-varied-v3-measurements-20261027/image-locators.json").is_file():
        pytest.skip("preserved corrected V3 artifacts are unavailable")
    safe = tmp_path / "safe-inputs.json"
    artifact = materializer.prepare_v3_safe_manifest(
        image_locators_path=repo_root / "f2-varied-v3-measurements-20261027/image-locators.json",
        downstream_validation_receipt_path=repo_root / "f2-varied-v3-downstream-v2-20261027/downstream-validation-receipt.json",
        request_bundle_path=repo_root / "f2-varied-v3-downstream-v2-20261027/requests/request-materialization-receipt.json",
        recovered_result_path=repo_root / "outputs/pipeline-20261028/rgb-mask-predictor-01a18128/approach-a-heldout-onepass-aa46da6-01a18128/recovered-publication/result.json",
        independent_review_path=repo_root / "outputs/pipeline-20261028/rgb-mask-predictor-01a18128/reviews/predictor-publish-recovery-27c6699-independent-review.md",
        output_path=safe,
    )
    drifted = Path(artifact["records"][0]["source_receipt_path"])
    original_sha256 = materializer._sha256

    def drift_one_source(path: Path) -> str:
        if path == drifted:
            return "0" * 64
        return original_sha256(path)

    monkeypatch.setattr(materializer, "_sha256", drift_one_source)
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        materializer.PredictedMaskMaterializationError,
        match="source renderer receipt SHA-256 mismatch",
    ):
        materializer.materialize_predictions(
            safe_manifest_path=safe,
            output_dir=output,
            materializer_commit="f" * 40,
        )
    assert not output.exists()

"""Tests for publishing accepted per-vial F2 receipts into stage manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(root: Path, vial_ordinal: int) -> Path:
    root.mkdir(parents=True)
    stage_steps = (1842, 3282, 4482)
    tanks = ("t200", "t400", "t100")
    pairs = []
    frame_paths = []
    frame_hashes = []
    mask_hashes = []
    for index, (step, tank) in enumerate(zip(stage_steps, tanks, strict=True), start=1):
        rgb = root / f"stage-{index}-{tank}.png"
        mask = root / f"stage-{index}-{tank}-mask.png"
        pixels = np.full((480, 640, 3), (20 * vial_ordinal, 30 * index, 80), np.uint8)
        liquid = np.zeros((480, 640), np.uint8)
        liquid[300:400, 285:335] = 255
        Image.fromarray(pixels, mode="RGB").save(rgb)
        Image.fromarray(liquid, mode="L").save(mask)
        frame_paths.append(rgb.name)
        frame_hashes.append(_sha(rgb))
        mask_hashes.append(_sha(mask))
        pairs.append(
            {
                "physics_step": step,
                "rgb_path": rgb.name,
                "rgb_sha256": _sha(rgb),
                "mask_path": mask.name,
                "mask_sha256": _sha(mask),
            }
        )
    payload = {
        "schema_version": 20,
        "experiment": "physx_three_pour_f2_paint_ball_emitter",
        "render_path": "point_instancer_sphere_emitter",
        "status": "pass",
        "versions": {"isaac": "6.2.0.0", "kit": "110.4.0", "physx": "110.4.4"},
        "provenance": {
            "script_sha256": "a" * 64,
            "requested_stage_sha256": "b" * 64,
        },
        "qualification": {"first_stage_only": False},
        "plan": {
            "pours": [
                {
                    "ordinal": ordinal,
                    "tank_id": tank,
                    "programmed_volume_ml": 20.0,
                    "end_step": end_step,
                    "stage_capture_step": stage_step,
                }
                for ordinal, (tank, end_step, stage_step) in enumerate(
                    zip(tanks, (1842, 3282, 4446), stage_steps, strict=True), start=1
                )
            ]
        },
        "recipe8_leading_residual": {
            "model_input_record": {
                "current_recipe_id": "video5-recipe-8",
                "previous_recipe_id": "video5-recipe-7",
                "dispense_order": list(tanks),
                "programmed_ml": {"t100": 20.0, "t200": 20.0, "t400": 20.0},
            }
        },
        "render": {
            "resolution": [640, 480],
            "stage_frame_physics_steps": list(stage_steps),
            "stage_frame_paths": frame_paths,
            "stage_frame_sha256": frame_hashes,
        },
        "semantics": {"pairs": pairs},
        "timing": {
            "physics_hz": 120,
            "pour_end_steps": [1842, 3282, 4446],
            "primary_final_capture_step": 4482,
            "primary_final_offset_s": 0.3,
        },
    }
    path = root / "receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _request(tmp_path: Path) -> dict[str, object]:
    vials = []
    for ordinal, vial_id in enumerate(("V1", "V2", "V3"), start=1):
        receipt = _receipt(tmp_path / vial_id, ordinal)
        vials.append(
            {
                "vial_id": vial_id,
                "receipt_path": str(receipt),
                "receipt_sha256": _sha(receipt),
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "f2_stage_batch_publish_request",
        "data_origin": "twin",
        "provider": "brev",
        "runtime_name": "NVIDIA Isaac Sim",
        "runtime_id": "gmufqeanr",
        "family_id": "FAMILY-RECIPE8-QUAL",
        "batch_id": "BATCH-RECIPE8-QUAL",
        "run_id": "RUN-RECIPE8-QUAL-01",
        "cohort": "scored_target",
        "current_recipe_id": "video5-recipe-8",
        "previous_recipe_id": "video5-recipe-7",
        "split_role": "val",
        "split_version": "inn-7-splits-v2",
        "programmed_recipe": {
            "dispense_order": ["t200", "t400", "t100"],
            "previous_ml_by_tank": {"t100": 20, "t200": 20, "t400": 20},
            "current_ml_by_tank": {"t100": 20, "t200": 20, "t400": 20},
        },
        "vials": vials,
    }


def test_publish_three_receipts_creates_bindable_nine_row_manifest_and_context(
    tmp_path, monkeypatch
):
    import fluidmix.f2_receipt_manifest_publisher as publisher
    from fluidmix.capture_manifest_adapter import bind_stage_capture_manifest

    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: {"status": "pass", "verified_artifact_count": 9, "failures": []},
    )
    output = tmp_path / "published"

    record_path = publisher.publish_f2_stage_batch(_request(tmp_path), output)

    assert record_path == output / "publication-receipt.json"
    manifest_path = output / "f2-stage-capture-manifest.json"
    context_path = output / "stage-batch-context.json"
    runtime_path = output / "f2-runtime-acceptance-receipt.json"
    assert {path.name for path in output.iterdir()} == {
        record_path.name,
        manifest_path.name,
        context_path.name,
        runtime_path.name,
    }
    manifest = json.loads(manifest_path.read_text())
    context = json.loads(context_path.read_text())
    assert len(manifest["rows"]) == 9
    assert {row["vial_id"] for row in manifest["rows"]} == {"V1", "V2", "V3"}
    assert [event["tank_id"] for event in manifest["stage_events"]] == [
        "t200",
        "t400",
        "t100",
    ]
    assert context["batches"][0]["batch_id"] == "BATCH-RECIPE8-QUAL"
    assert context["batches"][0]["cohort"] == "scored_target"
    assert not any(
        token in json.dumps({"manifest": manifest, "context": context}).lower()
        for token in ("residual_volume", "severity", "expected_disposition", "label_provenance")
    )
    bound = bind_stage_capture_manifest(manifest_path)
    assert bound["denominators"] == {"image_measurements": 9, "stage_statistics": 9}


def _passing_artifacts(_path):
    return {"status": "pass", "verified_artifact_count": 9, "failures": []}


def _rewrite_receipt(request, vial_index, mutation):
    entry = request["vials"][vial_index]
    path = Path(entry["receipt_path"])
    value = json.loads(path.read_text())
    mutation(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    entry["receipt_sha256"] = _sha(path)


def test_declared_receipt_hash_mismatch_fails_before_publication(tmp_path, monkeypatch):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    Path(request["vials"][0]["receipt_path"]).write_text("{}\n")
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="receipt_hash_mismatch"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")

    assert not (tmp_path / "published").exists()


def test_public_paint_ball_validator_is_mandatory(tmp_path):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="receipt_substantive_validation_failed",
    ):
        publisher.publish_f2_stage_batch(
            _request(tmp_path),
            tmp_path / "published",
        )


def test_official_artifact_validator_is_mandatory(tmp_path, monkeypatch):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="receipt_artifact_validation_failed",
    ):
        publisher.publish_f2_stage_batch(_request(tmp_path), tmp_path / "published")


@pytest.mark.parametrize(
    "mutation,artifact_result,reason",
    [
        (
            lambda receipt: receipt.update(status="insufficient_evidence"),
            _passing_artifacts(None),
            "receipt_not_accepted_full_run",
        ),
        (
            lambda _receipt: None,
            {
                "status": "insufficient_evidence",
                "verified_artifact_count": 8,
                "failures": ["artifact:stage-mask:missing"],
            },
            "receipt_artifact_validation_failed",
        ),
    ],
)
def test_nonpass_receipt_or_missing_artifact_gate_fails_closed(
    tmp_path, monkeypatch, mutation, artifact_result, reason
):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    _rewrite_receipt(request, 0, mutation)
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(
        publisher,
        "_official_artifact_validator",
        lambda _path: artifact_result,
    )

    with pytest.raises(publisher.F2ReceiptPublicationError, match=reason):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_missing_receipt_declared_stage_pair_fails_closed(tmp_path, monkeypatch):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    _rewrite_receipt(request, 1, lambda receipt: receipt["semantics"]["pairs"].pop(1))
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="receipt_stage_pair_missing"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


@pytest.mark.parametrize("field", ["current_recipe_id", "run_id"])
def test_receipt_recipe_or_run_context_mismatch_fails_closed(
    tmp_path, monkeypatch, field
):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    request = _request(tmp_path)

    def mutate(receipt):
        record = receipt["recipe8_leading_residual"]["model_input_record"]
        record[field] = "WRONG"

    _rewrite_receipt(request, 2, mutate)
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="receipt_context_mismatch"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_duplicate_vial_id_is_rejected(tmp_path, monkeypatch):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    request["vials"][2]["vial_id"] = "V2"
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(publisher.F2ReceiptPublicationError, match="vial_ids_invalid"):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_one_receipt_cannot_be_reused_for_two_vials(tmp_path, monkeypatch):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    request["vials"][2]["receipt_path"] = request["vials"][1]["receipt_path"]
    request["vials"][2]["receipt_sha256"] = request["vials"][1][
        "receipt_sha256"
    ]
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    with pytest.raises(
        publisher.F2ReceiptPublicationError,
        match="vial_receipts_not_distinct",
    ):
        publisher.publish_f2_stage_batch(request, tmp_path / "published")


def test_publication_is_no_clobber(tmp_path, monkeypatch):
    import fluidmix.f2_receipt_manifest_publisher as publisher

    request = _request(tmp_path)
    output = tmp_path / "published"
    monkeypatch.setattr(publisher, "validate_paint_ball_receipt", lambda _value: ())
    monkeypatch.setattr(publisher, "_official_artifact_validator", _passing_artifacts)

    publisher.publish_f2_stage_batch(request, output)
    before = {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }
    with pytest.raises(FileExistsError):
        publisher.publish_f2_stage_batch(request, output)

    assert before == {
        path.name: path.read_bytes() for path in output.iterdir() if path.is_file()
    }


def test_publisher_cli_is_help_safe():
    script = ROOT / "scripts" / "publish_f2_stage_batch.py"

    completed = subprocess.run(
        [sys.executable, "-S", str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--request" in completed.stdout
    assert "--output-dir" in completed.stdout


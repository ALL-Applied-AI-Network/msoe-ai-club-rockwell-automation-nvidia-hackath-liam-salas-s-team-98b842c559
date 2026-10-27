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


def _manifest(
    tmp_path: Path,
    *,
    batch_id: str = "BATCH-1",
    family_id: str = "VA",
    rgb_value: tuple[int, int, int] = (20, 40, 80),
    unique_rgb_content: bool = False,
) -> dict[str, object]:
    pixel_dir = tmp_path / "pixels"
    pixel_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for vial in ("V1", "V2", "V3"):
        for ordinal, (stage_id, time_code) in enumerate(
            (("after-t200", 171), ("after-t400", 342), ("after-t100", 686)),
            start=1,
        ):
            rgb_path = pixel_dir / f"{vial}-{stage_id}-rgb.png"
            mask_path = pixel_dir / f"{vial}-{stage_id}-mask.png"
            rgb = np.full((480, 640, 3), rgb_value, dtype=np.uint8)
            if unique_rgb_content:
                # Keep the measured liquid pixel unchanged while ensuring a
                # valid-decision fixture does not exercise the duplicate guard.
                rgb[-1, -1] = np.frombuffer(
                    hashlib.sha256(
                        f"{batch_id}:{vial}:{stage_id}".encode("utf-8")
                    ).digest()[:3],
                    dtype=np.uint8,
                )
            Image.fromarray(
                rgb,
                mode="RGB",
            ).save(rgb_path)
            Image.fromarray(
                np.pad(
                    np.array([[0, 255], [0, 0]], dtype=np.uint8),
                    ((0, 478), (0, 638)),
                    constant_values=0,
                ),
                mode="L",
            ).save(mask_path)
            rows.append(
                {
                    "vial_id": vial,
                    "stage_id": stage_id,
                    "stage_ordinal": ordinal,
                    "time_code": time_code,
                    "view_id": "vial_fill",
                    "verified": True,
                    "rgb": {"path": str(rgb_path), "sha256": _sha(rgb_path)},
                    "liquid_mask": {
                        "path": str(mask_path),
                        "sha256": _sha(mask_path),
                    },
                }
            )
    return {
        "schema_version": 1,
        "artifact_type": "twin_stage_capture_manifest",
        "status": "verified",
        "data_origin": "twin",
        "family_id": family_id,
        "batch_id": batch_id,
        "recipe_id": "S0380",
        "split_role": "val",
        "split_version": "inn-7-splits-v2",
        "view_id": "vial_fill",
        "stage_slots": [
            {"stage_id": "after-t200", "stage_ordinal": 1, "time_code": 171},
            {"stage_id": "after-t400", "stage_ordinal": 2, "time_code": 342},
            {"stage_id": "after-t100", "stage_ordinal": 3, "time_code": 686},
        ],
        "rows": rows,
    }


def _model_plan(target_batch: str, reference_batches: list[str]) -> dict[str, object]:
    stages = []
    for ordinal, stage_id in enumerate(("after-t200", "after-t400", "after-t100"), start=1):
        target_id = f"{target_batch}-V1-{stage_id}"
        primary_case = {
            "case_id": "all_reference_batches",
            "mode": "all_reference_batches",
            "alternative_id": None,
            "candidate_peers": [
                {
                    "peer_image_id": f"{batch}-V2-{stage_id}",
                    "batch_id": batch,
                    "origin": "reference_bank",
                }
                for batch in reference_batches
            ],
            "disposition_source": True,
        }
        alternative_cases = [
            {
                "case_id": "same_batch_only",
                "mode": "same_batch_only",
                "alternative_id": None,
                "candidate_peers": [
                    {
                        "peer_image_id": f"{target_batch}-V{vial}-{stage_id}",
                        "batch_id": target_batch,
                        "origin": "later_same_target_batch",
                    }
                    for vial in (2, 3)
                ],
                "disposition_source": False,
            }
        ]
        alternative_cases.extend(
            {
                "case_id": f"one_reference_batch:{batch}",
                "mode": "one_reference_batch",
                "alternative_id": batch,
                "candidate_peers": [
                    {
                        "peer_image_id": f"{batch}-V{vial}-{stage_id}",
                        "batch_id": batch,
                        "origin": "reference_bank",
                    }
                    for vial in (1, 2, 3)
                ],
                "disposition_source": False,
            }
            for batch in reference_batches
        )
        stages.append(
            {
                "evidence_id": f"EVID-{target_id}",
                "measurement_id": f"MEAS-{target_id}",
                "stage_id": stage_id,
                "stage_ordinal": ordinal,
                "time_code": (171, 342, 686)[ordinal - 1],
                "target_image_id": target_id,
                "reference_cases": [*alternative_cases, primary_case],
                "primary_mode": "all_reference_batches",
                "unit": "delta_e76",
            }
        )
    return {
        "schema_version": 3,
        "artifact_type": "model_input_plan",
        "config_id": "capture-adapter-test",
        "decision_records": [
            {
                "decision_id": f"DEC-{target_batch}-V1",
                "family_id": "VA",
                "batch_id": target_batch,
                "cohort": "scored_target",
                "partition": "val",
                "previous_sim_recipe_id": "S0160",
                "current_sim_recipe_id": "S0380",
                "previous_programmed_ml": {"t100": 30, "t200": 20, "t400": 10},
                "current_programmed_ml": {"t100": 10, "t200": 30, "t400": 20},
                "programmed_dispense_order": ["t200", "t400", "t100"],
                "vial_id": "V1",
                "vial_position": 1,
                "stages": stages,
                "decision_timing": "retrospective_after_complete_three_vial_batch_no_latency_claim",
            }
        ],
        "real_data_used": False,
    }


def test_verified_manifest_binds_nine_png_measurements_and_stage_statistics(tmp_path):
    from fluidmix.capture_manifest_adapter import bind_stage_capture_manifest

    result = bind_stage_capture_manifest(_manifest(tmp_path))

    assert result["artifact_type"] == "stage_capture_measurements"
    assert result["data_origin"] == "twin"
    assert result["batch_id"] == "BATCH-1"
    assert result["recipe_id"] == "S0380"
    assert result["split_role"] == "val"
    assert len(result["image_locators"]["images"]) == 9
    assert len(result["image_measurements"]) == 9
    assert all(row["validity"] == "valid" for row in result["image_measurements"])
    assert len(result["stage_statistics"]) == 9
    assert all(isinstance(row, dict) for row in result["stage_statistics"])
    assert {row["vial_id"] for row in result["stage_statistics"]} == {"V1", "V2", "V3"}
    assert {row["stage_ordinal"] for row in result["stage_statistics"]} == {1, 2, 3}
    model_view = json.dumps(result["model_inputs"], sort_keys=True)
    assert "rgb_path" not in model_view
    assert "mask_path" not in model_view
    assert "semantic_aov" not in model_view
    json.dumps(result, sort_keys=True)


def test_bound_statistics_feed_existing_decision_boundary_as_insufficient_same_batch(tmp_path):
    from fluidmix.capture_manifest_adapter import (
        bind_stage_capture_manifest,
        stage_statistics_from_result,
    )
    from fluidmix.stage_decision import (
        FrozenStageDecisionConfig,
        StageReference,
        decide_vial_stages,
    )

    stats = stage_statistics_from_result(bind_stage_capture_manifest(_manifest(tmp_path)))
    targets = [row for row in stats if row.vial_id == "V1"]
    refs = {
        target.stage_id: [
            StageReference(row, "later_same_batch")
            for row in stats
            if row.stage_id == target.stage_id and row.vial_id in {"V2", "V3"}
        ]
        for target in targets
    }
    result = decide_vial_stages(
        "DEC-BATCH-1-V1",
        targets,
        refs,
        FrozenStageDecisionConfig("test", 2, 2, 2.0, 2.0, 1.0),
    )
    assert result["disposition"] == "insufficient_evidence"
    assert "independent_reference_batches_below_minimum" in result["limit_codes"]


def test_bound_locators_feed_existing_cosmos_image_input_validator(tmp_path):
    from fluidmix.capture_manifest_adapter import bind_stage_capture_manifest
    from fluidmix.stage_cosmos_inputs import _checked_images

    result = bind_stage_capture_manifest(_manifest(tmp_path))
    first = next(iter(result["image_locators"]["images"]))
    spec = {
        "condition_id": "ordered_stage_visual_pairs",
        "image_ids": [first],
    }
    checked = _checked_images(
        spec,
        image_locators=result["image_locators"]["images"],
        config={
            "max_images_per_attempt": 6,
            "image_width": 640,
            "image_height": 480,
        },
    )
    assert checked == [
        {
            "image_id": first,
            "path": result["image_locators"]["images"][first]["rgb_path"],
            "sha256": result["image_locators"]["images"][first]["rgb_sha256"],
        }
    ]


@pytest.mark.parametrize(
    ("reference_count", "target_rgb", "expected"),
    [
        (2, (0, 0, 0), "keep"),
        (2, (255, 255, 255), "discard"),
        (1, (0, 0, 0), "insufficient_evidence"),
    ],
)
def test_assembler_publishes_canonical_cosmos_evidence(
    tmp_path, reference_count, target_rgb, expected
):
    from fluidmix.capture_manifest_adapter import (
        bind_stage_capture_manifest,
        build_measured_evidence,
    )
    from fluidmix.stage_cosmos_explanation import stage_reference_response

    target_batch = "TARGET"
    ref_batches = [f"REF-{index}" for index in range(1, reference_count + 1)]
    captures = [
        bind_stage_capture_manifest(
            _manifest(
                tmp_path / target_batch,
                batch_id=target_batch,
                rgb_value=target_rgb,
                unique_rgb_content=True,
            )
        )
    ]
    captures.extend(
        bind_stage_capture_manifest(
            _manifest(
                tmp_path / batch,
                batch_id=batch,
                rgb_value=(0, 0, 0),
                unique_rgb_content=True,
            )
        )
        for batch in ref_batches
    )

    measured = build_measured_evidence(
        _model_plan(target_batch, ref_batches), captures
    )

    assert measured["artifact_type"] == "measured_evidence"
    assert measured["decision_records"][0]["disposition"] == expected
    response = stage_reference_response(measured["decision_records"][0])
    assert response.startswith("Evidence:")


def test_assembler_rejects_duplicate_batches_and_unreferenced_images(tmp_path):
    from fluidmix.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
        build_measured_evidence,
    )

    target = bind_stage_capture_manifest(_manifest(tmp_path / "target", batch_id="TARGET"))
    refs = bind_stage_capture_manifest(_manifest(tmp_path / "ref", batch_id="REF-1"))
    plan = _model_plan("TARGET", ["REF-1"])
    with pytest.raises(CaptureManifestError) as caught:
        build_measured_evidence(plan, [target, target])
    assert caught.value.reason_code == "duplicate_capture_batch"
    plan_with_unreferenced_rows = _model_plan("TARGET", ["REF-1"])
    for stage in plan_with_unreferenced_rows["decision_records"][0]["stages"]:
        stage["reference_cases"] = [stage["reference_cases"][-1]]
    with pytest.raises(CaptureManifestError) as caught:
        build_measured_evidence(plan_with_unreferenced_rows, [target, refs])
    assert caught.value.reason_code == "capture_image_ids_mismatch"


def test_assembler_rejects_capture_provenance_drift(tmp_path):
    from fluidmix.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
        build_measured_evidence,
    )

    target = bind_stage_capture_manifest(_manifest(tmp_path / "target", batch_id="TARGET"))
    drifted = bind_stage_capture_manifest(
        _manifest(tmp_path / "ref", batch_id="REF-1", family_id="TE_HIGH")
    )
    with pytest.raises(CaptureManifestError) as caught:
        build_measured_evidence(_model_plan("TARGET", ["REF-1"]), [target, drifted])
    assert caught.value.reason_code == "capture_measurement_provenance_mismatch"


def test_assembler_rejects_forged_peer_batch_declaration(tmp_path):
    from fluidmix.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
        build_measured_evidence,
    )

    target = bind_stage_capture_manifest(_manifest(tmp_path / "target", batch_id="TARGET"))
    reference = bind_stage_capture_manifest(_manifest(tmp_path / "ref", batch_id="REF-1"))
    plan = _model_plan("TARGET", ["REF-1"])
    for stage in plan["decision_records"][0]["stages"]:
        for case in stage["reference_cases"]:
            for peer in case["candidate_peers"]:
                if peer["peer_image_id"].startswith("REF-1-"):
                    peer["batch_id"] = "REF-FORGED"
    with pytest.raises(CaptureManifestError) as caught:
        build_measured_evidence(plan, [target, reference])
    assert caught.value.reason_code == "capture_measurement_provenance_mismatch"


def test_assembler_rejects_cross_recipe_reference_declaration(tmp_path):
    from fluidmix.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
        build_measured_evidence,
    )

    target = bind_stage_capture_manifest(_manifest(tmp_path / "target", batch_id="TARGET"))
    reference = bind_stage_capture_manifest(_manifest(tmp_path / "ref", batch_id="REF-1"))
    plan = _model_plan("TARGET", ["REF-1"])
    for stage in plan["decision_records"][0]["stages"]:
        for case in stage["reference_cases"]:
            for peer in case["candidate_peers"]:
                if peer["peer_image_id"].startswith("REF-1-"):
                    peer["current_sim_recipe_id"] = "S9999"
    with pytest.raises(CaptureManifestError) as caught:
        build_measured_evidence(plan, [target, reference])
    assert caught.value.reason_code == "model_input_peer_provenance_mismatch"


def test_plan_image_provenance_allows_distinct_family_contexts():
    from fluidmix.capture_manifest_adapter import _plan_image_provenance

    def decision(prefix, family, recipe, partition):
        return {
            "family_id": family,
            "current_sim_recipe_id": recipe,
            "partition": partition,
            "batch_id": f"{prefix}-TARGET",
            "vial_id": "V1",
            "stages": [
                {
                    "stage_id": "STAGE1",
                    "stage_ordinal": 1,
                    "target_image_id": f"{prefix}-TARGET-IMAGE",
                    "reference_cases": [
                        {
                            "candidate_peers": [
                                {
                                    "peer_image_id": f"{prefix}-REFERENCE-IMAGE",
                                    "batch_id": f"{prefix}-REFERENCE",
                                }
                            ]
                        }
                    ],
                }
            ],
        }

    contexts = _plan_image_provenance(
        [
            decision("VA", "VA", "S0380", "val"),
            decision("TE", "TE_HIGH", "S0431", "test"),
        ]
    )
    assert contexts["VA-TARGET-IMAGE"]["family_id"] == "VA"
    assert contexts["TE-TARGET-IMAGE"]["family_id"] == "TE_HIGH"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda manifest: manifest["rows"].pop(), "missing_stage_row"),
        (lambda manifest: manifest["rows"].append(manifest["rows"][0]), "duplicate_stage_row"),
        (lambda manifest: manifest["rows"][0].update(view_id="tank_level"), "wrong_view_id"),
        (lambda manifest: manifest["rows"][0].update(time_code=172), "wrong_time_code"),
        (lambda manifest: manifest["rows"][0].update(verified=False), "row_unverified"),
    ],
)
def test_manifest_rows_fail_closed_with_reason_codes(tmp_path, change, reason):
    from fluidmix.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
    )

    manifest = _manifest(tmp_path)
    change(manifest)
    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == reason


def test_manifest_rejects_non_png_and_hash_drift(tmp_path):
    from fluidmix.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path)
    manifest["rows"][0]["rgb"]["path"] = manifest["rows"][0]["liquid_mask"]["path"]
    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "rgb_hash_mismatch"

    manifest = _manifest(tmp_path / "other")
    mask = Path(manifest["rows"][0]["liquid_mask"]["path"])
    mask.write_bytes(b"tampered")
    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "mask_hash_mismatch"


def test_manifest_rejects_non_png_suffix_and_empty_mask(tmp_path):
    from fluidmix.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path)
    rgb = Path(manifest["rows"][0]["rgb"]["path"])
    non_png = rgb.with_suffix(".jpg")
    non_png.write_bytes(rgb.read_bytes())
    manifest["rows"][0]["rgb"] = {"path": str(non_png), "sha256": _sha(non_png)}
    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "rgb_not_png"

    manifest = _manifest(tmp_path / "empty")
    mask = Path(manifest["rows"][0]["liquid_mask"]["path"])
    Image.fromarray(np.zeros((480, 640), dtype=np.uint8), mode="L").save(mask)
    manifest["rows"][0]["liquid_mask"]["sha256"] = _sha(mask)
    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "empty_mask"


def test_manifest_rejects_non_fillcam_resolution(tmp_path):
    from fluidmix.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path)
    rgb = Path(manifest["rows"][0]["rgb"]["path"])
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(rgb)
    manifest["rows"][0]["rgb"]["sha256"] = _sha(rgb)
    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "rgb_resolution_invalid"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("split_role", "validation", "split_role_invalid"),
        ("split_version", "old-splits", "split_version_invalid"),
    ],
)
def test_manifest_requires_frozen_twin_partition_provenance(tmp_path, field, value, reason):
    from fluidmix.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path)
    manifest[field] = value
    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == reason


def test_manifest_path_resolves_relative_artifacts_and_preserves_hash_binding(tmp_path):
    from fluidmix.capture_manifest_adapter import bind_stage_capture_manifest

    manifest = _manifest(tmp_path)
    for row in manifest["rows"]:
        for key in ("rgb", "liquid_mask"):
            artifact = row[key]
            artifact["path"] = str(Path(artifact["path"]).relative_to(tmp_path))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = bind_stage_capture_manifest(manifest_path)
    assert all(
        Path(row["rgb_path"]).is_absolute()
        for row in result["image_locators"]["images"].values()
    )

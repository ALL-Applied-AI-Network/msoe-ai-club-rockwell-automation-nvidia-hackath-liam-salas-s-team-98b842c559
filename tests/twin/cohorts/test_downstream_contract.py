"""Regression contract for canonical F2 capture -> plan -> measurement routing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_binding() -> dict[str, object]:
    return {
        "provider": "brev",
        "runtime_name": "NVIDIA Isaac Sim",
        "isaac_sim_version": "6.2.0.0",
        "runtime_id": "brev-isaac62-f2-accepted-v1",
        "source_sha256": "b" * 64,
    }


def _runtime_receipt(root: Path) -> dict[str, str]:
    path = root / "accepted-f2-runtime-receipt.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "f2_runtime_acceptance_receipt",
        "status": "accepted",
        "provider": "brev",
        "runtime_id": "brev-isaac62-f2-accepted-v1",
        "isaac_sim_version": "6.2.0.0",
        "source_sha256": "b" * 64,
        "gates": {
            "scene_parity": True,
            "finite_particle_state": True,
            "containment": True,
            "zero_leakage": True,
            "volume_height": True,
            "stream_continuity": True,
            "source_color": True,
            "numeric_mixing": True,
            "visual_mixing": True,
            "semantics_alignment": True,
            "timing": True,
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "sha256": _sha(path)}


def _varied_runtime_receipt(root: Path) -> dict[str, str]:
    path = root / "accepted-f2-varied-runtime-receipt.json"
    hard_gates = {
        "scene_parity",
        "finite_particle_state",
        "containment",
        "zero_leakage",
        "rendered_body_volume_height",
        "stream_continuity",
        "source_color",
        "numeric_mixing",
        "visual_mixing",
        "semantics_alignment",
        "timing",
    }
    payload = {
        "schema_version": 3,
        "artifact_type": "f2_varied_runtime_acceptance_receipt",
        "status": "accepted",
        "provider": "brev",
        "runtime_id": "brev-isaac62-f2-accepted-v1",
        "isaac_sim_version": "6.2.0.0",
        "source_sha256": "b" * 64,
        "hard_gates": {name: True for name in sorted(hard_gates)},
        "particle_height_diagnostic": {
            "hard_gate": False,
            "physics_pass_claimed": False,
            "status": "within_bounds",
            "bounds_passed": True,
            "warning_count": 0,
            "warned_vial_count": 0,
            "vial_count": 3,
            "claim": "particle_height_within_bounds_diagnostic_only",
            "vials": [
                {
                    "case_id": f"CASE_{ordinal:016x}",
                    "vial_id": f"V{ordinal}",
                    "status": "within_bounds",
                    "bounds_passed": True,
                    "warning_count": 0,
                }
                for ordinal in (1, 2, 3)
            ],
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "sha256": _sha(path)}


def _events() -> list[dict[str, object]]:
    return [
        {
            "event_id": f"dispense-{ordinal}-{tank}",
            "stage_ordinal": ordinal,
            "tank_id": tank,
            "tank_order": ordinal,
            "physics_step": step,
            "frame_id": f"frame-{step}",
            "event_relative_timestamp_s": float(ordinal * 10),
            "final_observation_offset_s": 1.0,
        }
        for ordinal, (tank, step) in enumerate(
            (("t200", 640), ("t400", 1280), ("t100", 3840)), start=1
        )
    ]


def _manifest(
    root: Path,
    batch_id: str,
    *,
    run_id: str | None = None,
    rgb_value: tuple[int, int, int] = (20, 40, 80),
) -> dict[str, object]:
    root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    events = _events()
    batch_code = sum(batch_id.encode("utf-8")) % 97 + 1
    for vial_index, vial_id in enumerate(("V1", "V2", "V3"), start=1):
        for event in events:
            ordinal = int(event["stage_ordinal"])
            stem = f"{vial_id}-{event['event_id']}"
            rgb_path = root / f"{stem}-rgb.png"
            mask_path = root / f"{stem}-mask.png"
            rgb = np.full((480, 640, 3), rgb_value, dtype=np.uint8)
            # Unique unmasked nuisance pixel proves artifact reuse is not hidden.
            rgb[479, 639] = (batch_code, vial_index, ordinal)
            mask = np.zeros((480, 640), dtype=np.uint8)
            left = batch_code + 20 * vial_index + ordinal
            mask[100:300, left : left + 80 + ordinal] = 255
            Image.fromarray(rgb, mode="RGB").save(rgb_path)
            Image.fromarray(mask, mode="L").save(mask_path)
            rows.append(
                {
                    "vial_id": vial_id,
                    "event_id": event["event_id"],
                    "stage_ordinal": ordinal,
                    "physics_step": event["physics_step"],
                    "frame_id": event["frame_id"],
                    "event_relative_timestamp_s": event["event_relative_timestamp_s"],
                    "final_observation_offset_s": event["final_observation_offset_s"],
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
        "schema_version": 2,
        "artifact_type": "f2_stage_capture_manifest",
        "status": "accepted",
        "data_origin": "twin",
        "family_id": "VA",
        "batch_id": batch_id,
        "run_id": run_id or f"run-{batch_id}",
        "recipe_id": "S0380",
        "split_role": "val",
        "split_version": "inn-7-splits-v2",
        "view_id": "vial_fill",
        "runtime_binding": _runtime_binding(),
        "runtime_receipt": _runtime_receipt(root),
        "stage_events": events,
        "rows": rows,
    }


def _context(names: tuple[str, ...], run_ids: dict[str, str] | None = None) -> dict:
    run_ids = run_ids or {name: f"run-{name}" for name in names}
    return {
        "schema_version": 1,
        "artifact_type": "stage_batch_context",
        "split_version": "inn-7-splits-v2",
        "batches": [
            {
                "batch_id": name,
                "cohort": "scored_target" if name == "TARGET" else "reference_only",
                "family_id": "VA",
                "current_recipe_id": "S0380",
                "previous_recipe_id": "S0160",
                "run_id": run_ids[name],
                "split_role": "val",
                "programmed_recipe": {
                    "dispense_order": ["t200", "t400", "t100"],
                    "previous_ml_by_tank": {"t200": 20, "t400": 10, "t100": 30},
                    "current_ml_by_tank": {"t200": 30, "t400": 20, "t100": 10},
                },
            }
            for name in names
        ],
    }


def _pipeline(tmp_path: Path, names: tuple[str, ...], *, run_ids=None):
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        bind_stage_capture_manifest,
        build_measured_evidence,
    )
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    paths: list[Path] = []
    manifests: list[dict[str, object]] = []
    for name in names:
        manifest = _manifest(
            tmp_path / name,
            name,
            run_id=(run_ids or {}).get(name),
        )
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)
        manifests.append(manifest)
    plan = build_model_input_plan(paths, _context(names, run_ids))
    captures = [bind_stage_capture_manifest(row) for row in manifests]
    return plan, captures, build_measured_evidence(plan, captures)


def test_varied_v3_capture_binds_two_active_stages_without_family_id(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest

    manifest = _manifest(
        tmp_path / "varied",
        "BATCH-VARIED-2-STAGE",
        run_id="RUN-VALIDATION-01",
    )
    manifest["schema_version"] = 3
    manifest.pop("family_id")
    manifest["recipe_id"] = "13"
    manifest["split_role"] = "validation"
    manifest["split_version"] = "whole-run-varied-v3"
    manifest["runtime_receipt"] = _varied_runtime_receipt(tmp_path / "varied")
    active_event_ids = {
        event["event_id"] for event in manifest["stage_events"][:2]
    }
    manifest["stage_events"] = manifest["stage_events"][:2]
    manifest["rows"] = [
        row for row in manifest["rows"] if row["event_id"] in active_event_ids
    ]

    bound = bind_stage_capture_manifest(manifest)

    assert bound["capture_contract_version"] == "f2-brev-isaac62-varied-event-v3"
    assert bound["split_role"] == "validation"
    assert bound["split_version"] == "whole-run-varied-v3"
    assert bound["denominators"] == {
        "image_measurements": 6,
        "stage_statistics": 6,
    }
    assert len(bound["stage_events"]) == 2
    assert "family_id" not in bound


def test_varied_v4_capture_uses_the_same_strict_event_schema_with_v4_split(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest

    manifest = _manifest(
        tmp_path / "varied-v4",
        "BATCH-V4-2-STAGE",
        run_id="RUN-V4-VALIDATION-01",
    )
    manifest["schema_version"] = 3
    manifest.pop("family_id")
    manifest["recipe_id"] = "13"
    manifest["split_role"] = "validation"
    manifest["split_version"] = "whole-run-varied-v4"
    manifest["runtime_receipt"] = _varied_runtime_receipt(tmp_path / "varied-v4")
    active_event_ids = {
        event["event_id"] for event in manifest["stage_events"][:2]
    }
    manifest["stage_events"] = manifest["stage_events"][:2]
    manifest["rows"] = [
        row for row in manifest["rows"] if row["event_id"] in active_event_ids
    ]

    bound = bind_stage_capture_manifest(manifest)

    assert bound["capture_contract_version"] == "f2-brev-isaac62-varied-event-v3"
    assert bound["split_version"] == "whole-run-varied-v4"
    assert bound["denominators"] == {
        "image_measurements": 6,
        "stage_statistics": 6,
    }


def test_dynamic_per_image_mask_hashes_are_valid_with_two_external_runs(tmp_path):
    from fluidmix.explanation.template.stage_cosmos_explanation import stage_reference_response

    plan, captures, measured = _pipeline(tmp_path, ("TARGET", "REF-A", "REF-B"))

    mask_hashes = {
        locator["mask_sha256"]
        for capture in captures
        for locator in capture["image_locators"]["images"].values()
    }
    assert len(mask_hashes) == 27
    assert plan["schema_version"] == 2
    assert all(
        set(capture["runtime_binding"])
        == {"runtime_id", "runtime_receipt_sha256"}
        for capture in captures
    )
    assert "accepted-f2-runtime-receipt.json" not in json.dumps(captures)
    assert "scene_parity" not in json.dumps(captures)
    assert "runtime_receipt" not in json.dumps(plan)
    assert "runtime_receipt" not in json.dumps(measured)
    assert {row["disposition"] for row in measured["decision_records"]} == {"keep"}
    assert all("time_code" not in stage for row in plan["decision_records"] for stage in row["stages"])
    assert stage_reference_response(measured["decision_records"][0]).startswith("Evidence:")


def test_target_context_plus_one_external_batch_is_insufficient(tmp_path):
    _, _, measured = _pipeline(tmp_path, ("TARGET", "REF-A"))

    assert {row["disposition"] for row in measured["decision_records"]} == {
        "insufficient_evidence"
    }
    assert measured["denominators"]["abstained_vial_decisions"] == 3


def test_two_external_batches_from_one_external_run_are_insufficient(tmp_path):
    names = ("TARGET", "REF-A", "REF-B")
    _, _, measured = _pipeline(
        tmp_path,
        names,
        run_ids={"TARGET": "target-run", "REF-A": "external-run", "REF-B": "external-run"},
    )

    assert {row["disposition"] for row in measured["decision_records"]} == {
        "insufficient_evidence"
    }


def test_context_cannot_forge_external_run_identity(tmp_path):
    from fluidmix.explanation.vlm.model_input_plan_builder import ModelInputPlanError, build_model_input_plan

    names = ("TARGET", "REF-A", "REF-B")
    paths = []
    for name in names:
        manifest = _manifest(tmp_path / name, name)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)
    context = _context(names)
    context["batches"][1]["run_id"] = "forged-external-run"

    with pytest.raises(ModelInputPlanError, match="context_capture_run_mismatch"):
        build_model_input_plan(paths, context)


def test_predicted_mask_quality_failure_propagates_and_abstains(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest, build_measured_evidence
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    manifests = [
        _manifest(tmp_path / name, name) for name in ("TARGET", "REF-A", "REF-B")
    ]
    row = manifests[0]["rows"][0]
    mask_path = Path(row["liquid_mask"]["path"])
    Image.fromarray(np.zeros((480, 640), dtype=np.uint8), mode="L").save(mask_path)
    row["liquid_mask"]["sha256"] = _sha(mask_path)

    paths = []
    for manifest in manifests:
        path = tmp_path / f"{manifest['batch_id']}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)
    plan = build_model_input_plan(paths, _context(("TARGET", "REF-A", "REF-B")))
    captures = [bind_stage_capture_manifest(manifest) for manifest in manifests]
    bound = captures[0]

    first = next(item for item in bound["image_measurements"] if item["image_id"].endswith("V1-dispense-1-t200"))
    assert first["validity"] == "invalid"
    assert first["limit_codes"] == ["mask_empty"]
    measured = build_measured_evidence(plan, captures)
    target_v1 = next(row for row in measured["decision_records"] if row["vial_id"] == "V1")
    assert target_v1["disposition"] == "insufficient_evidence"
    assert "mask_empty" in target_v1["limit_codes"]


def test_only_accepted_brev_isaac62_runtime_binds_new_f2_rows(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path / "target", "TARGET")
    manifest["runtime_binding"]["isaac_sim_version"] = "6.0.1.0"

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "f2_runtime_binding_invalid"


def test_forged_runtime_receipt_hash_stops_binding(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path / "target", "TARGET")
    manifest["runtime_receipt"]["sha256"] = "a" * 64

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "runtime_receipt_hash_mismatch"


def test_false_substantive_runtime_gate_stops_binding(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path / "target", "TARGET")
    receipt_path = Path(manifest["runtime_receipt"]["path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["gates"]["volume_height"] = False
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    manifest["runtime_receipt"]["sha256"] = _sha(receipt_path)

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "runtime_receipt_gate_failed"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda path: path.unlink(), "runtime_receipt_missing"),
        (lambda path: path.write_bytes(b"not-json"), "runtime_receipt_unreadable"),
    ],
)
def test_missing_or_unreadable_runtime_receipt_stops_binding(tmp_path, mutation, reason):
    from fluidmix.storage.artifacts.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path / "target", "TARGET")
    receipt_path = Path(manifest["runtime_receipt"]["path"])
    mutation(receipt_path)
    if receipt_path.exists():
        manifest["runtime_receipt"]["sha256"] = _sha(receipt_path)

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == reason


def test_artifact_hash_failure_stops_f2_binding(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path / "target", "TARGET")
    Path(manifest["rows"][0]["rgb"]["path"]).write_bytes(b"tampered")

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "rgb_hash_mismatch"


def test_capture_local_row_frame_ids_bind_to_shared_physics_step_events(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest

    manifest = _manifest(tmp_path / "target", "TARGET")
    for event in manifest["stage_events"]:
        event["frame_id"] = None
    for row_number, row in enumerate(manifest["rows"], start=1):
        row["frame_id"] = f"capture-{row_number:04d}"

    bound = bind_stage_capture_manifest(manifest)

    assert [event["frame_id"] for event in bound["stage_events"]] == [
        None,
        None,
        None,
    ]
    assert bound["denominators"] == {
        "image_measurements": 9,
        "stage_statistics": 9,
    }


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda row: row.pop("frame_id"), "row_malformed"),
        (lambda row: row.update(frame_id=""), "frame_id_invalid"),
    ],
)
def test_missing_or_empty_capture_local_frame_id_fails_closed(
    tmp_path, mutation, reason
):
    from fluidmix.storage.artifacts.capture_manifest_adapter import (
        CaptureManifestError,
        bind_stage_capture_manifest,
    )

    manifest = _manifest(tmp_path / "target", "TARGET")
    mutation(manifest["rows"][0])

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == reason


def test_stage_event_mismatch_fails_closed(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import CaptureManifestError, bind_stage_capture_manifest

    manifest = _manifest(tmp_path / "target", "TARGET")
    for event in manifest["stage_events"]:
        event["frame_id"] = None
    for row_number, row in enumerate(manifest["rows"], start=1):
        row["frame_id"] = f"capture-{row_number:04d}"
    manifest["rows"][0]["physics_step"] += 1

    with pytest.raises(CaptureManifestError) as caught:
        bind_stage_capture_manifest(manifest)
    assert caught.value.reason_code == "stage_event_mismatch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("composition", {"t100": 1.0}),
        ("appearance_blend_alpha", 0.3),
        ("particle_count", 3840),
        ("expected_volume_ml", 60),
        ("exposure", 1.5),
        ("nuisance_seed", 42),
        ("expected_disposition", "keep"),
    ],
)
def test_central_deploy_allowlist_rejects_f2_oracles(tmp_path, field, value):
    from fluidmix.explanation.vlm.model_input_plan_builder import ModelInputPlanError, build_model_input_plan

    names = ("TARGET", "REF-A", "REF-B")
    paths = []
    for name in names:
        manifest = _manifest(tmp_path / name, name)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)
    context = _context(names)
    context["batches"][0][field] = value

    with pytest.raises(ModelInputPlanError, match="forbidden_deploy_input_field"):
        build_model_input_plan(paths, context)


def test_f2_decision_config_is_hash_bound(tmp_path):
    from fluidmix.storage.artifacts.capture_manifest_adapter import bind_stage_capture_manifest, build_measured_evidence
    from fluidmix.explanation.vlm.model_input_plan_builder import build_model_input_plan

    names = ("TARGET", "REF-A", "REF-B")
    paths = []
    captures = []
    for name in names:
        manifest = _manifest(tmp_path / name, name)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(path)
        captures.append(bind_stage_capture_manifest(manifest))
    plan = build_model_input_plan(paths, _context(names))
    plan["decision_config"]["threshold_margin_delta_e76"] = 9.0

    with pytest.raises(ValueError, match="decision config hash mismatch"):
        build_measured_evidence(plan, captures)

    encoded = json.dumps(
        plan["decision_config"], sort_keys=True, separators=(",", ":")
    ).encode()
    plan["decision_config_sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="differs from frozen behavior"):
        build_measured_evidence(plan, captures)


def test_public_clis_route_f2_events_through_canonical_measurement(tmp_path):
    names = ("TARGET", "REF-A", "REF-B")
    manifests = []
    for name in names:
        manifest = _manifest(tmp_path / name, name)
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        manifests.append(path)
    context = tmp_path / "context.json"
    context.write_text(json.dumps(_context(names)), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_command = [
        sys.executable,
        str(ROOT / "cli/fluidmix/explanation/vlm" / "build_model_input_plan_from_manifests.py"),
    ]
    for manifest in manifests:
        plan_command.extend(("--capture-manifest", str(manifest)))
    plan_command.extend(("--batch-context", str(context), "--out", str(plan_path)))
    planned = subprocess.run(
        plan_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert planned.returncode == 0, planned.stderr

    evidence_path = tmp_path / "measured.json"
    locator_path = tmp_path / "locators.json"
    evidence_command = [
        sys.executable,
        str(ROOT / "cli/fluidmix/measurement" / "build_stage_measured_evidence.py"),
        "--model-input-plan",
        str(plan_path),
    ]
    for manifest in manifests:
        evidence_command.extend(("--capture-manifest", str(manifest)))
    evidence_command.extend(
        ("--out", str(evidence_path), "--image-locators-out", str(locator_path))
    )
    measured = subprocess.run(
        evidence_command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    assert measured.returncode == 0, measured.stderr
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert plan["schema_version"] == 2
    assert "time_code" not in json.dumps(plan)
    assert {row["disposition"] for row in evidence["decision_records"]} == {"keep"}

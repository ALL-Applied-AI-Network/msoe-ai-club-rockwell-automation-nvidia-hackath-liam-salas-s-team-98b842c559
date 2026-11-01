"""End-to-end diagnostic bundle for the Varied V3 evidence path."""

from __future__ import annotations

from copy import deepcopy
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
SCRIPT = ROOT / "cli/fluidmix/pipeline" / "build_f2_varied_demo.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _stage(batch: str, vial: str, ordinal: int, tank: str) -> dict[str, object]:
    stage = f"dispense-{ordinal}-{tank}"
    peers = [candidate for candidate in ("V1", "V2", "V3") if candidate != vial]
    return {
        "target_evidence_id": f"EVID-{batch}-{vial}-{stage}",
        "target_measurement_id": f"MEAS-{batch}-{vial}-{stage}",
        "target_image_id": f"{batch}-{vial}-{stage}",
        "stage_id": stage,
        "stage_ordinal": ordinal,
        "tank_id": tank,
        "relative_status": "outlier_to_two_peer_mode",
        "peer_pair_id": f"PAIR-{batch}-{stage}-{peers[0]}-{peers[1]}",
        "peer_evidence_ids": [f"EVID-{batch}-{peer}-{stage}" for peer in peers],
        "peer_measurement_ids": [f"MEAS-{batch}-{peer}-{stage}" for peer in peers],
        "peer_image_ids": [f"{batch}-{peer}-{stage}" for peer in peers],
        "target_to_peer_pairs": [
            {
                "pair_id": f"PAIR-{batch}-{stage}-{vial}-{peer}",
                "delta_e76": 6.0 + index,
            }
            for index, peer in enumerate(peers)
        ],
        "peer_pair_delta_e76": 0.5,
        "reference_center_lab": [40.0, -2.0, 3.0],
        "reference_radius_delta_e76": 0.25,
        "target_delta_e76": 6.5,
        "threshold_delta_e76": 1.25,
        "threshold_excess_delta_e76": 5.25,
        "unit": "delta_e76",
        "limit_codes": ["relative_outlier_direction_unresolved"],
    }


def _decision(run_index: int, vial_index: int) -> dict[str, object]:
    batch = f"BATCH_{run_index:016x}"
    vial = f"V{vial_index}"
    tanks = ["t200", "t400", "t100"] if run_index in {1, 4} else ["t200", "t100"]
    return {
        "decision_id": f"DEC-{batch}-{vial}",
        "batch_id": batch,
        "run_id": f"RUN-{run_index:02d}",
        "recipe_id": str(run_index),
        "split_role": (
            "train"
            if run_index <= 4
            else ("validation" if run_index == 5 else "evaluation")
        ),
        "split_version": "whole-run-varied-v3",
        "vial_id": vial,
        "disposition": "insufficient_evidence",
        "explanation_status": "UNKNOWN",
        "relative_assessment": "outlier_to_two_peer_mode",
        "stages": [
            _stage(batch, vial, ordinal, tank)
            for ordinal, tank in enumerate(tanks, 1)
        ],
        "limit_codes": [
            "within_run_reference_cleanliness_unverified",
            "relative_outlier_not_directional",
        ],
        "decision_timing": (
            "retrospective_after_complete_three_vial_batch_no_latency_claim"
        ),
    }


def _decisions() -> dict[str, object]:
    rows = [_decision(run, vial) for run in range(1, 7) for vial in range(1, 4)]
    return {
        "schema_version": 1,
        "artifact_type": "f2_varied_triad_decisions",
        "status": "assessed",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
        "source_measurement_sha256": "a" * 64,
        "config": {
            "config_id": "v3-relative-diagnostic-inherited-stage-v1",
            "tight_peer_pair_max_delta_e76": 2.0,
            "target_outlier_margin_delta_e76": 1.0,
        },
        "config_sha256": "b" * 64,
        "reference_policy": {
            "scope": "other_two_vials_same_run_same_stage",
            "position_prior": False,
            "cleanliness_verified": False,
            "authorization": "relative_evidence_only_no_keep_or_discard",
        },
        "decision_records": rows,
        "denominators": {
            "runs": 6,
            "vial_decisions": 18,
            "keep": 0,
            "discard": 0,
            "insufficient_evidence": 18,
            "relative_outlier_vials": 18,
        },
        "claim_boundary": (
            "Synthetic within-run colour relations only. No verified clean reference."
        ),
    }


def _write_images(root: Path, decisions: dict[str, object]) -> dict[str, object]:
    images: dict[str, object] = {}
    image_ids = {
        image_id
        for decision in decisions["decision_records"]
        for stage in decision["stages"]
        for image_id in [stage["target_image_id"], *stage["peer_image_ids"]]
    }
    for index, image_id in enumerate(sorted(image_ids)):
        parts = image_id.split("-")
        vial_index = next(i for i, part in enumerate(parts) if part in {"V1", "V2", "V3"})
        batch = "-".join(parts[:vial_index])
        vial = parts[vial_index]
        stage_id = "-".join(parts[vial_index + 1 :])
        ordinal = int(stage_id.split("-")[1])
        tank = stage_id.split("-")[2]
        matching = next(
            row
            for row in decisions["decision_records"]
            if row["batch_id"] == batch and row["vial_id"] == vial
        )
        rgb_path = root / f"rgb-{index:02d}.png"
        mask_path = root / f"mask-{index:02d}.png"
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[..., 0] = 20 + index
        rgb[..., 1] = 80
        rgb[..., 2] = 140
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[260:400, 292:334] = 255
        Image.fromarray(rgb, mode="RGB").save(rgb_path)
        Image.fromarray(mask, mode="L").save(mask_path)
        images[image_id] = {
            "batch_id": batch,
            "mask_path": str(mask_path.resolve()),
            "mask_sha256": _sha(mask_path),
            "recipe_id": matching["recipe_id"],
            "rgb_path": str(rgb_path.resolve()),
            "rgb_sha256": _sha(rgb_path),
            "run_id": matching["run_id"],
            "source_manifest_sha256": "c" * 64,
            "split_role": matching["split_role"],
            "split_version": "whole-run-varied-v3",
            "stage_id": stage_id,
            "stage_ordinal": ordinal,
            "tank_id": tank,
            "vial_id": vial,
        }
    assert len(images) == 42
    return {
        "schema_version": 1,
        "artifact_type": "f2_varied_image_locators",
        "status": "published",
        "data_origin": "twin",
        "real_data_used": False,
        "split_version": "whole-run-varied-v3",
        "source_manifests": [],
        "source_validation_receipt": {"path": "/unused", "sha256": "d" * 64},
        "images": images,
        "denominators": {"images": 42, "vials": 18},
        "claim_boundary": "Synthetic renderer image locations only.",
    }


def _score(decisions: dict[str, object], decision_sha: str) -> dict[str, object]:
    from fluidmix.explanation.template.varied_explanation import (
        parse_varied_response,
        structure_varied_response,
        varied_reference_response,
    )

    structured = []
    for decision in decisions["decision_records"]:
        raw = varied_reference_response(decision)
        record = structure_varied_response(raw, parse_varied_response(raw), decision)
        structured.append(
            {
                "attempt_id": f"deterministic_template-{decision['decision_id']}",
                "decision_id": decision["decision_id"],
                "condition_id": "deterministic_template",
                "kind": "control",
                "record": record,
            }
        )
    passing = {"passed": 18, "total": 18}
    return {
        "schema_version": 1,
        "artifact_type": "f2_varied_cosmos_nano_score",
        "status": "scored",
        "decision_artifact_sha256": decision_sha,
        "config_sha256": "e" * 64,
        "raw_results": {"path": "/unused", "sha256": "f" * 64},
        "score": {
            "schema_version": "f2_varied_cosmos_score.v1",
            "matrix": {
                "planned": 72,
                "completed": 72,
                "errors": 0,
                "model_completed": 54,
                "control_completed": 18,
            },
            "metrics": {
                "lenient_parse_control": passing,
                "template_control_content_policy": passing,
                "required_abstention_control": passing,
            },
            "condition_metrics": {
                "deterministic_template": {
                    "completed": 18,
                    "lenient_parse": 18,
                    "strict_content_policy": 18,
                }
            },
            "failures": [],
            "structured_records": structured,
            "claim_boundary": "Synthetic explanation quality only.",
        },
    }


@pytest.fixture()
def demo_inputs(tmp_path: Path) -> dict[str, Path]:
    decisions = _decisions()
    decisions_path = tmp_path / "decisions.json"
    _write_json(decisions_path, decisions)
    locators_path = tmp_path / "locators.json"
    _write_json(locators_path, _write_images(tmp_path, decisions))
    score_path = tmp_path / "score.json"
    _write_json(score_path, _score(decisions, _sha(decisions_path)))
    return {
        "decisions": decisions_path,
        "locators": locators_path,
        "score": score_path,
        "out": tmp_path / "demo",
    }


def _build(paths: dict[str, Path]):
    from fluidmix.pipeline.varied_demo import build_varied_demo_bundle

    return build_varied_demo_bundle(
        paths["decisions"], paths["locators"], paths["score"], paths["out"]
    )


def test_bundle_is_atomic_oracle_diagnostic_with_all_abstentions(demo_inputs):
    receipt = _build(demo_inputs)

    assert sorted(path.name for path in demo_inputs["out"].iterdir()) == [
        "REVIEW.md",
        "operator-records.json",
        "receipt.json",
    ]
    records = json.loads(
        (demo_inputs["out"] / "operator-records.json").read_text(encoding="utf-8")
    )
    assert records["deployment_eligible"] is False
    assert records["mask_source"] == "renderer_semantic_mask_oracle_diagnostic"
    assert records["limit_codes"] == ["rgb_only_predicted_liquid_mask_missing"]
    assert records["denominators"] == {
        "decisions": 18,
        "images": 42,
        "insufficient_evidence": 18,
    }
    assert len(records["operator_records"]) == 18
    assert {row["disposition"] for row in records["operator_records"]} == {
        "insufficient_evidence"
    }
    assert all(
        "rgb_only_predicted_liquid_mask_missing" in row["limit_codes"]
        for row in records["operator_records"]
    )
    assert (
        records["operator_records"][0]["deterministic_explanation"]["response_raw"]
        == "Evidence: UNKNOWN | The supplied within-run evidence cannot establish a verified reference.\n"
        "Uncertainty: ABSTAIN | Cause, physical defect, and next-vial effect remain unverified.\n"
        "Action: COLLECT_EVIDENCE | Collect an independent verified reference before assigning a disposition."
    )
    review = (demo_inputs["out"] / "REVIEW.md").read_text(encoding="utf-8")
    assert "deployment eligible: **no**" in review.lower()
    assert str(next(iter(json.loads(demo_inputs["locators"].read_text())["images"].values()))["rgb_path"]) in review
    assert receipt["denominators"] == {"decisions": 18, "images": 42}
    assert receipt == json.loads(
        (demo_inputs["out"] / "receipt.json").read_text(encoding="utf-8")
    )


def test_public_cli_builds_the_same_diagnostic_boundary(demo_inputs):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--decisions",
            str(demo_inputs["decisions"]),
            "--image-locators",
            str(demo_inputs["locators"]),
            "--cosmos-score",
            str(demo_inputs["score"]),
            "--out",
            str(demo_inputs["out"]),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "18 abstaining vial decisions" in completed.stdout
    receipt = json.loads(
        (demo_inputs["out"] / "receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["deployment_eligible"] is False


def test_existing_destination_fails_without_changing_it(demo_inputs):
    demo_inputs["out"].mkdir()
    marker = demo_inputs["out"] / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    from fluidmix.pipeline.varied_demo import VariedDemoError

    with pytest.raises(VariedDemoError, match="destination_exists"):
        _build(demo_inputs)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert sorted(path.name for path in demo_inputs["out"].iterdir()) == ["keep.txt"]


def test_rgb_byte_tamper_fails_before_publication(demo_inputs):
    locators = json.loads(demo_inputs["locators"].read_text(encoding="utf-8"))
    first = next(iter(locators["images"].values()))
    Path(first["rgb_path"]).write_bytes(b"tampered")
    from fluidmix.pipeline.varied_demo import VariedDemoError

    with pytest.raises(VariedDemoError, match="rgb_hash_mismatch"):
        _build(demo_inputs)
    assert not demo_inputs["out"].exists()


def test_structured_template_tamper_fails_before_publication(demo_inputs):
    score = json.loads(demo_inputs["score"].read_text(encoding="utf-8"))
    score["score"]["structured_records"][0]["record"]["presentation"][
        "suggested_action"
    ] = "keep"
    _write_json(demo_inputs["score"], score)
    from fluidmix.pipeline.varied_demo import VariedDemoError

    with pytest.raises(VariedDemoError, match="template_record_drift"):
        _build(demo_inputs)
    assert not demo_inputs["out"].exists()


@pytest.mark.parametrize(
    "forbidden_key",
    ["oracle", "label", "residual_volume", "residual_volume_ml", "palette"],
)
def test_forbidden_input_fields_fail_closed(demo_inputs, forbidden_key):
    decisions = json.loads(demo_inputs["decisions"].read_text(encoding="utf-8"))
    decisions[forbidden_key] = "hidden"
    _write_json(demo_inputs["decisions"], decisions)
    score = json.loads(demo_inputs["score"].read_text(encoding="utf-8"))
    score["decision_artifact_sha256"] = _sha(demo_inputs["decisions"])
    _write_json(demo_inputs["score"], score)
    from fluidmix.pipeline.varied_demo import VariedDemoError

    with pytest.raises(VariedDemoError, match="forbidden_input_field"):
        _build(demo_inputs)
    assert not demo_inputs["out"].exists()


def test_model_structured_record_and_failed_template_metric_are_rejected(demo_inputs):
    score = json.loads(demo_inputs["score"].read_text(encoding="utf-8"))
    model = deepcopy(score["score"]["structured_records"][0])
    model["attempt_id"] = "numbers_only-" + model["decision_id"]
    model["condition_id"] = "numbers_only"
    model["kind"] = "model"
    score["score"]["structured_records"].append(model)
    _write_json(demo_inputs["score"], score)
    from fluidmix.pipeline.varied_demo import VariedDemoError

    with pytest.raises(VariedDemoError, match="model_structured_record_forbidden"):
        _build(demo_inputs)

    score["score"]["structured_records"].pop()
    score["score"]["metrics"]["required_abstention_control"]["passed"] = 17
    _write_json(demo_inputs["score"], score)
    with pytest.raises(VariedDemoError, match="template_metric_not_passing"):
        _build(demo_inputs)


def test_raced_empty_destination_is_not_replaced(demo_inputs, monkeypatch):
    import fluidmix.pipeline.varied_demo as demo

    original = demo._rename_directory_no_replace

    def race(source, destination):
        destination.mkdir()
        original(source, destination)

    monkeypatch.setattr(demo, "_rename_directory_no_replace", race)

    with pytest.raises(demo.VariedDemoError, match="destination_exists"):
        _build(demo_inputs)

    assert demo_inputs["out"].is_dir()
    assert list(demo_inputs["out"].iterdir()) == []
    assert not list(demo_inputs["out"].parent.glob(".demo.tmp-*"))

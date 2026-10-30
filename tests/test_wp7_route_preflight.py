"""CPU-only contracts for selecting the dormant WP7 explanation route."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha(path)


def _vial(
    root: Path, *, vial: int, role: str, start_cursor: int, run: str = "RUN"
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]], dict[str, int]]:
    clip = root / "clips" / f"{run}-{vial}.mp4"
    clip_sha = _write(clip, f"synthetic-clip-{vial}".encode())
    records: list[dict[str, object]] = []
    locators: dict[str, dict[str, str]] = {}
    cursors: dict[str, int] = {}
    for stage_ordinal, tank in enumerate(("t100", "t200"), start=1):
        evidence_id = f"E-{run}-{vial}-{stage_ordinal}"
        image_id = f"I-{run}-{vial}-{stage_ordinal}"
        rgb = root / "rgb" / f"{image_id}.png"
        mask = root / "predicted" / f"{image_id}.png"
        rgb_sha = _write(rgb, f"synthetic-rgb-{evidence_id}".encode())
        mask_sha = _write(mask, f"synthetic-prediction-{evidence_id}".encode())
        records.append(
            {
                "run_id": run,
                "case_id": f"CASE-{vial}",
                "vial_ordinal": vial,
                "block_recipe_id": "BLOCK-R",
                "block_position": vial,
                "recipe_id": "R",
                "previous_recipe_id": "P" if vial == 1 else "R",
                "recipe_ml": {"t100": 1.0, "t200": 1.0, "t400": 0.0},
                "dispense_order": ["t100", "t200"],
                "split_role": role,
                "split_version": "synthetic-preflight-v1",
                "stage_id": f"after-{tank}",
                "tank_id": tank,
                "stage_ordinal": stage_ordinal,
                "fill_clip_path": clip.relative_to(root).as_posix(),
                "fill_frame_range": [vial * 10, vial * 10 + 5],
                "transition_frame_ranges": {"entry": None, "exit": None},
                "clip_sha256": clip_sha,
                "image_id": image_id,
                "rgb_sha256": rgb_sha,
                "mask_sha256": mask_sha,
                "mask_source": "predicted",
                "measurement_id": f"M-{run}-{vial}-{stage_ordinal}",
                "median_lab": [50.0, float(stage_ordinal), 0.0],
                "mad_lab": [1.0, 1.0, 1.0],
                "iqr_lab": [2.0, 2.0, 2.0],
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
        locators[image_id] = {
            "rgb_path": str(rgb),
            "rgb_sha256": rgb_sha,
            "mask_path": str(mask),
            "mask_sha256": mask_sha,
        }
        cursors[evidence_id] = start_cursor + stage_ordinal - 1
    return records, locators, cursors


def _history_query(
    tmp_path: Path, *, nonempty: bool = True, role: str = "train"
):
    from fluidmix.evidence_store import EvidenceStore

    artifact_root = tmp_path / ("history-artifacts" if nonempty else "empty-artifacts")
    batches = [
        _vial(artifact_root, vial=1, role=role, start_cursor=10),
        _vial(artifact_root, vial=2, role=role, start_cursor=20),
    ]
    records = [record for batch in batches for record in batch[0]]
    locators = {key: value for batch in batches for key, value in batch[1].items()}
    cursors = {key: value for batch in batches for key, value in batch[2].items()}
    store = EvidenceStore(
        tmp_path / ("history.sqlite" if nonempty else "empty-history.sqlite"),
        artifact_root=artifact_root,
    )
    store.ingest(records, locators, phase="measured", event_cursors=cursors)
    return store.vial_history(
        "RUN",
        before=20 if nonempty else 10,
        allowed_roles={role},
        target_evidence_id="E-RUN-2-1" if nonempty else "E-RUN-1-1",
    )


def _development_reference_queries(tmp_path: Path):
    from fluidmix.wp7_development_reference import (
        ACCEPTED_CNN_PREDICTOR,
        DevelopmentReferenceStore,
        _create_minimal_store,
    )

    records = []
    configurations = []
    cursor = 0
    for recipe_number in range(1, 9):
        recipe = f"R-{recipe_number:02d}"
        peers: dict[str, list[str]] = {}
        for batch, run in (
            ("INN67-DEV-RB-B01", "DONOR-A"),
            ("INN67-DEV-RB-B02", "DONOR-B"),
        ):
            for ordinal, tank in enumerate(("t100", "t200"), start=1):
                stage = f"after-{tank}"
                evidence_id = f"E-{batch}-{recipe}-{ordinal}"
                image_id = f"I-{batch}-{recipe}-{ordinal}"
                rgb = tmp_path / "artifacts" / "rgb" / f"{image_id}.png"
                mask = tmp_path / "artifacts" / "mask" / f"{image_id}.png"
                records.append({
                    "evidence_id": evidence_id, "event_cursor": cursor,
                    "batch_id": batch, "run_id": run, "case_id": f"CASE-{recipe}-{batch}",
                    "recipe_id": recipe, "stage_id": stage, "stage_ordinal": ordinal,
                    "tank_id": tank, "split_role": "development_reference",
                    "image_id": image_id, "rgb_path": str(rgb),
                    "rgb_sha256": _write(rgb, f"rgb:{evidence_id}".encode()),
                    "mask_path": str(mask),
                    "mask_sha256": _write(mask, f"mask:{evidence_id}".encode()),
                    "median_lab": [50.0, float(ordinal), 0.0],
                    "source_render_receipt_sha256": "1" * 64,
                    "source_validation_receipt_sha256": "2" * 64,
                    "source_render_plan_sha256": "3" * 64,
                    "known_good_profile_sha256": "47bc0b6f483a121a28871b3eb74ec27efb79996b83d03d8c45d6f5b0b9cdeb7c",
                })
                peers.setdefault(stage, []).append(evidence_id)
                cursor += 1
        for ordinal, tank in enumerate(("t100", "t200"), start=1):
            stage = f"after-{tank}"
            configurations.append({
                "recipe_id": recipe, "stage_id": stage, "stage_ordinal": ordinal,
                "tank_id": tank, "peer_evidence_ids": peers[stage],
                "threshold_delta_e76": 1.5,
                "configuration_target_evidence_id": f"CONFIG-{recipe}-{ordinal}",
            })
    database = tmp_path / "development-reference.sqlite3"
    _create_minimal_store(database, records=records, configurations=configurations,
        provenance={
            "development_reference_bank_sha256": "9" * 64,
            "delta_e_frozen_config_sha256": "a" * 64,
            "materialization_receipt_sha256": "b" * 64,
            "predicted_store_export_sha256": "c" * 64,
            "derived_source_manifest_sha256": "d" * 64,
            "predictor": ACCEPTED_CNN_PREDICTOR,
            "predictor_sha256": _canonical_hash(ACCEPTED_CNN_PREDICTOR),
            "accepted_store_tuple": {
                "database_path": "/accepted/inn67/evidence.sqlite3",
                "database_sha256": "f" * 64,
                "population_receipt_path": "/accepted/inn67/population-receipt.json",
                "population_receipt_sha256": "0" * 64,
            },
            "inn67_provenance": {"fixed": "test"},
        })
    snapshot = hashlib.sha256(database.read_bytes()).hexdigest()
    return DevelopmentReferenceStore(
        database, expected_snapshot_sha256=snapshot
    ).coverage_queries()


def _mask_provenance() -> dict[str, object]:
    return {
        "schema_version": "wp7-predicted-mask-provenance-v1",
        "artifact_type": "predicted_mask_provenance",
        "source_class": "reviewed_rgb_only_prediction",
        "split_role": "test_and_real_check",
        "source_rgb_sha256": "1" * 64,
        "mask_sha256": "2" * 64,
        "runtime_id": "reviewed-runtime-v1",
        "runtime_sha256": "3" * 64,
        "config_sha256": "4" * 64,
        "validation_receipt_sha256": "5" * 64,
        "accepts_real_rgb_only": True,
    }


def _evaluation_store() -> dict[str, object]:
    return {
        "schema_version": "wp7-evaluation-only-store-v1",
        "artifact_type": "evaluation_only_store_provenance",
        "store_purpose": "evaluation_only",
        "store_schema_version": 3,
        "store_schema_sha256": "a" * 64,
        "store_snapshot_sha256": "b" * 64,
        "review_receipt_sha256": "c" * 64,
        "stage_row_count": 0,
    }


def _history_store(query_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "wp7-production-history-store-v1",
        "artifact_type": "production_history_store_provenance",
        "store_purpose": "production_history",
        "store_schema_version": 3,
        "store_schema_sha256": "6" * 64,
        "store_snapshot_sha256": "7" * 64,
        "approval_receipt_sha256": "8" * 64,
        "source_classes": ["camera_rgb", "predicted_mask"],
        "query_sha256s": [query_sha256],
    }


def _route_decision() -> dict[str, object]:
    return {
        "schema_version": "wp7-route-decision-v1",
        "artifact_type": "wp7_route_decision",
        "issue_id": "INN-62",
        "scope": "section_6_history_backed_explanations",
        "approved_mode": "history_backed",
        "decision_status": "approved",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
    }


def _development_store(
    query_sha256s: list[str], *, snapshot_sha256: str = "7" * 64
) -> dict[str, object]:
    from fluidmix.wp7_development_reference import ACCEPTED_CNN_PREDICTOR

    predictor_sha = _canonical_hash(ACCEPTED_CNN_PREDICTOR)
    accepted_store = {
        "database_path": "/accepted/inn67/evidence.sqlite3",
        "database_sha256": "f" * 64,
        "population_receipt_path": "/accepted/inn67/population-receipt.json",
        "population_receipt_sha256": "0" * 64,
    }
    return {
        "schema_version": "wp7-development-reference-store-v1",
        "artifact_type": "development_reference_store_provenance",
        "status": "published",
        "store_purpose": "development_reference",
        "store_schema_version": 1,
        "store_schema_sha256": "6" * 64,
        "store_snapshot_sha256": snapshot_sha256,
        "development_reference_bank_sha256": "9" * 64,
        "delta_e_frozen_config_sha256": "a" * 64,
        "materialization_receipt_sha256": "b" * 64,
        "predicted_store_export_sha256": "c" * 64,
        "derived_source_manifest_sha256": "d" * 64,
        "predictor": ACCEPTED_CNN_PREDICTOR,
        "predictor_sha256": predictor_sha,
        "accepted_store_tuple": accepted_store,
        "inn67_provenance": {
            "reference_bank_plan_sha256": "f956b6d1a4b3f14fd57df1004594c330b5322f8ff5c32afe00c1aef895719b79",
            "reference_bank_inventory_sha256": "0b373388c040d0d01d6b8eb45f353b8cbfdeba5b5608f02a9ec5eda09f5e7f95",
            "pre_access_amendment_file_sha256": "230ca2c7dbfef1023d9e314fc40b5da70b6b1bdd59e184d1642540213f55d9a8",
            "pre_access_amendment_receipt_sha256": "3cc4212f84a6a011b0d3dd579cb6cf817285d3a946598908756f6c95c9f349db",
            "materialization_protocol_amendment_sha256": "15604a5251ae920c140ef5068f7057dc2ab4f1e7f1e2b7946408eef05e09d61c",
            "render_inventory_sha256": "1" * 64,
            "predicted_store_export_sha256": "c" * 64,
            "accepted_store_tuple": accepted_store,
        },
        "frozen_config_identity": {
            "config_id": "v3-relative-diagnostic-inherited-stage-v1",
            "distance_metric": "CIE76", "unit": "delta_e76",
            "tight_peer_pair_max_delta_e76": 2.0,
            "target_outlier_margin_delta_e76": 1.0,
            "decision_rule": "peer_delta = dE76(B01,B02); threshold = peer_delta/2 + 1.0; require peer_delta <= 2.0 and B03 target_to_center_delta_e76 <= threshold",
            "threshold_formula": "threshold_delta_e76 = peer_pair_delta_e76 / 2 + 1.0",
            "configuration_target_gate": "all 16 B03 recipe-stage targets must be consistent_with_two_peer_mode",
            "selection_input_batch_ids": ["INN67-DEV-RB-B03"],
            "reference_geometry_batch_ids": ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"],
            "configuration_target_batch_id": "INN67-DEV-RB-B03",
        },
        "source_classes": ["synthetic_rgb", "predicted_mask"],
        "coverage_query_sha256s": query_sha256s,
        "coverage_member_count": 32,
        "configuration_count": 16,
        "stage_row_count": 32,
        "donor_batch_ids": ["INN67-DEV-RB-B01", "INN67-DEV-RB-B02"],
        "authorization": {
            "issue_id": "INN-64", "option": 3, "approver_name": "Liam",
            "scope": "one_shot_after_accepted_development_reference_bank_and_frozen_config",
        },
        "one_shot_inputs_opened": False,
        "claim_boundary": "synthetic reference evidence only; not plant or physical truth",
    }


def _development_mask_provenance(store: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "wp7-development-predicted-mask-provenance-v1",
        "artifact_type": "development_reference_predicted_mask_provenance",
        "source_class": "accepted_rgb_only_cnn_predicted_masks",
        "predictor": store["predictor"],
        "predictor_sha256": store["predictor_sha256"],
        "accepted_store_tuple": store["accepted_store_tuple"],
        "predicted_store_export_sha256": store["predicted_store_export_sha256"],
    }


def _ordered_video_provenance() -> dict[str, object]:
    return {
        "schema_version": "wp7-ordered-test-video-binding-v1",
        "artifact_type": "ordered_test_video_binding",
        "ordered_video_path": "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-ordered-test-video-01a18128-attempt01/wp7-ordered-test-video.mp4",
        "ordered_video_sha256": "1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b",
        "vial_count": 18,
        "composition_receipt_path": "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-ordered-test-video-01a18128-attempt01/wp7-ordered-test-video.json",
        "composition_receipt_sha256": "0a88c47515af9fb0e389900f32504994914fe48bcc1e06bc392185382116b037",
        "final_review_path": "/home/hackathon5/liamreporoot/outputs/pipeline-20261028/wp7-ordered-test-video-final-review-01a18128-attempt01/FINAL-REVIEW.json",
        "final_review_sha256": "ae4f9b2fdc3c6341a77dfa25044b673d7357155a98795ffa0f2df8b93ddec683",
    }


def _development_route_decision() -> dict[str, object]:
    return {
        "schema_version": "wp7-development-reference-route-decision-v1",
        "artifact_type": "wp7_development_reference_route_decision",
        "issue_id": "INN-64",
        "option": 3,
        "scope": "wp1_and_video5_single_pass_with_development_reference_bank",
        "approved_mode": "development_reference_backed",
        "decision_status": "approved",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
        "evidence_boundary": "synthetic_reference_evidence_only_not_plant_or_physical_truth",
    }


def test_development_reference_route_is_distinct_hash_bound_and_synthetic(
    tmp_path: Path,
) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    queries = _development_reference_queries(tmp_path)
    decision = _development_route_decision()
    store = _development_store(
        [query.query_sha256 for query in queries],
        snapshot_sha256=queries[0].members[0].donor_store_snapshot_sha256,
    )
    receipt = preflight_explanation_route(
        requested_mode="development_reference_backed",
        expected_test_vials=18,
        expected_real_check_vials=30,
        development_reference_store_provenance=store,
        predicted_mask_provenance=_development_mask_provenance(store),
        ordered_video_provenance=_ordered_video_provenance(),
        history_queries=queries,
        route_decision=decision,
        expected_route_decision_sha256=_canonical_hash(decision),
    )

    assert receipt["selected_mode"] == "development_reference_backed"
    assert receipt["activation_status"] == "development_reference_requirements_satisfied"
    assert receipt["history"]["store_purpose"] == "development_reference"
    assert receipt["history"]["development_reference_bank_sha256"] == "9" * 64
    assert receipt["history"]["delta_e_frozen_config_sha256"] == "a" * 64
    assert receipt["history"]["materialization_receipt_sha256"] == "b" * 64
    assert receipt["evidence_boundary"] == (
        "synthetic_reference_evidence_only_not_plant_or_physical_truth"
    )
    assert receipt["test"]["vial_count"] == 18
    assert receipt["real_check"]["vial_count"] == 30
    assert receipt["predicted_mask"]["predictor"] == store["predictor"]
    assert receipt["history"]["accepted_store_tuple"] == store["accepted_store_tuple"]
    assert receipt["ordered_video"]["ordered_video_sha256"] == (
        "1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b"
    )

    import fluidmix.carryover_pipeline as pipeline

    assert pipeline._validate_execution_preflight(
        receipt,
        expected_sha256=str(receipt["preflight_sha256"]),
        selected_mode="development_reference_backed",
    ) == receipt
    with pytest.raises(ValueError, match="selected mode mismatch"):
        pipeline._validate_execution_preflight(
            receipt,
            expected_sha256=str(receipt["preflight_sha256"]),
            selected_mode="history_backed",
        )


def test_malformed_development_preflight_cannot_read_inputs_or_claim(
    tmp_path: Path, monkeypatch
) -> None:
    import fluidmix.carryover_pipeline as pipeline

    queries = _development_reference_queries(tmp_path / "query")
    decision = _development_route_decision()
    store = _development_store(
        [query.query_sha256 for query in queries],
        snapshot_sha256=queries[0].members[0].donor_store_snapshot_sha256,
    )
    receipt = pipeline.preflight_explanation_route(
        requested_mode="development_reference_backed",
        expected_test_vials=18,
        expected_real_check_vials=30,
        development_reference_store_provenance=store,
        predicted_mask_provenance=_development_mask_provenance(store),
        ordered_video_provenance=_ordered_video_provenance(),
        history_queries=queries,
        route_decision=decision,
        expected_route_decision_sha256=_canonical_hash(decision),
    )
    malformed = deepcopy(receipt)
    malformed["history"]["development_reference_bank_sha256"] = "0" * 64
    reads: list[Path] = []
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda path, _role: reads.append(Path(path)) or b"unexpected",
    )
    with pytest.raises(ValueError, match="preflight SHA-256 mismatch"):
        pipeline.run_test(
            video=tmp_path / "video.bin",
            recipe_sequence=tmp_path / "recipe.json",
            freeze=tmp_path / "freeze.json",
            expected_freeze_sha256="0" * 64,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
            preflight_receipt=malformed,
            expected_preflight_sha256=str(receipt["preflight_sha256"]),
            selected_mode="development_reference_backed",
        )
    assert reads == []
    assert not (tmp_path / "claim.json").exists()
    with pytest.raises(ValueError, match="ordered video SHA-256"):
        pipeline.run_test(
            video=Path(_ordered_video_provenance()["ordered_video_path"]),
            recipe_sequence=tmp_path / "recipe.json",
            freeze=tmp_path / "freeze.json",
            expected_freeze_sha256="0" * 64,
            store=Path(store["accepted_store_tuple"]["database_path"]),
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
            preflight_receipt=receipt,
            expected_preflight_sha256=str(receipt["preflight_sha256"]),
            selected_mode="development_reference_backed",
        )
    assert reads == [
        Path(_ordered_video_provenance()["ordered_video_path"]),
        Path(store["accepted_store_tuple"]["database_path"]),
        Path(store["accepted_store_tuple"]["population_receipt_path"]),
    ]
    assert not (tmp_path / "claim.json").exists()


def test_development_route_requires_exact_mask_video_and_store_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fluidmix.carryover_pipeline as pipeline

    queries = _development_reference_queries(tmp_path / "queries")
    store = _development_store(
        [query.query_sha256 for query in queries],
        snapshot_sha256=queries[0].members[0].donor_store_snapshot_sha256,
    )
    decision = _development_route_decision()
    common = {
        "requested_mode": "development_reference_backed",
        "expected_test_vials": 18,
        "expected_real_check_vials": 30,
        "development_reference_store_provenance": store,
        "history_queries": queries,
        "route_decision": decision,
        "expected_route_decision_sha256": _canonical_hash(decision),
        "ordered_video_provenance": _ordered_video_provenance(),
    }
    with pytest.raises(ValueError, match="predicted-mask provenance is required"):
        pipeline.preflight_explanation_route(**common)
    wrong_mask = _development_mask_provenance(store)
    wrong_mask["predictor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predictor"):
        pipeline.preflight_explanation_route(
            **common, predicted_mask_provenance=wrong_mask
        )
    receipt = pipeline.preflight_explanation_route(
        **common, predicted_mask_provenance=_development_mask_provenance(store)
    )
    reads: list[Path] = []
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda path, _role: reads.append(Path(path)) or b"unexpected",
    )
    with pytest.raises(ValueError, match="ordered video path identity"):
        pipeline.run_test(
            video=tmp_path / "wrong-video.mp4",
            recipe_sequence=tmp_path / "recipe.json",
            freeze=tmp_path / "freeze.json",
            expected_freeze_sha256="0" * 64,
            store=Path(store["accepted_store_tuple"]["database_path"]),
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
            preflight_receipt=receipt,
            expected_preflight_sha256=str(receipt["preflight_sha256"]),
            selected_mode="development_reference_backed",
        )
    assert reads == []
    assert not (tmp_path / "claim.json").exists()


def test_development_and_production_routes_reject_mixed_roles(tmp_path: Path) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    production_query = _history_query(tmp_path / "production")
    decision = _development_route_decision()
    with pytest.raises(TypeError, match="DevelopmentReferenceQuery"):
        store = _development_store(
            [production_query.query_sha256] + [f"{index:064x}" for index in range(1, 16)]
        )
        preflight_explanation_route(
            requested_mode="development_reference_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            development_reference_store_provenance=store,
            predicted_mask_provenance=_development_mask_provenance(store),
            ordered_video_provenance=_ordered_video_provenance(),
            history_queries=[production_query],
            route_decision=decision,
            expected_route_decision_sha256=_canonical_hash(decision),
        )

    development_queries = _development_reference_queries(tmp_path / "development")
    with pytest.raises(TypeError, match="HistoryQuery"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=_history_store(
                development_queries[0].query_sha256
            ),
            history_queries=[development_queries[0]],
        )


def test_evaluation_only_route_is_explicit_and_denominator_bound() -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    receipt = preflight_explanation_route(
        requested_mode="evaluation_only",
        expected_test_vials=18,
        expected_real_check_vials=30,
        evaluation_store_provenance=_evaluation_store(),
    )

    assert receipt["selected_mode"] == "evaluation_only"
    assert receipt["fallback_used"] is False
    assert receipt["test"] == {"terminal_behavior": "abstain", "vial_count": 18}
    assert receipt["real_check"] == {"terminal_behavior": "skip", "vial_count": 30}
    assert receipt["history_backed_explanations_enabled"] is False
    assert receipt["store"]["store_purpose"] == "evaluation_only"
    assert receipt["store"]["stage_row_count"] == 0
    assert len(receipt["preflight_sha256"]) == 64


def test_evaluation_only_rejects_ambiguous_history_inputs(tmp_path: Path) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    query = _history_query(tmp_path)
    with pytest.raises(ValueError, match="evaluation_only.*must not receive"):
        preflight_explanation_route(
            requested_mode="evaluation_only",
            expected_test_vials=18,
            expected_real_check_vials=30,
            evaluation_store_provenance=_evaluation_store(),
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=_history_store(query.query_sha256),
            history_queries=[query],
        )


def test_history_backed_never_falls_back_when_requirements_are_missing() -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    with pytest.raises(ValueError, match="predicted-mask provenance is required"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
        )


def test_evaluation_only_requires_a_reviewed_zero_row_store() -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    with pytest.raises(ValueError, match="evaluation-only store provenance is required"):
        preflight_explanation_route(
            requested_mode="evaluation_only",
            expected_test_vials=18,
            expected_real_check_vials=30,
        )

    nonempty = _evaluation_store()
    nonempty["stage_row_count"] = 1
    with pytest.raises(ValueError, match="stage_row_count"):
        preflight_explanation_route(
            requested_mode="evaluation_only",
            expected_test_vials=18,
            expected_real_check_vials=30,
            evaluation_store_provenance=nonempty,
        )

    wrong_schema = _evaluation_store()
    wrong_schema["store_schema_version"] = 2
    with pytest.raises(ValueError, match="store_schema_version"):
        preflight_explanation_route(
            requested_mode="evaluation_only",
            expected_test_vials=18,
            expected_real_check_vials=30,
            evaluation_store_provenance=wrong_schema,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"source_class": "oracle"}, "source_class"),
        ({"artifact_type": "sim_metadata"}, "artifact_type"),
        ({"split_role": "evaluation"}, "split_role"),
        ({"accepts_real_rgb_only": False}, "accepts_real_rgb_only"),
    ],
)
def test_history_backed_rejects_nonproduction_mask_provenance(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    query = _history_query(tmp_path)
    provenance = {**_mask_provenance(), **mutation}
    with pytest.raises(ValueError, match=message):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=provenance,
            history_store_provenance=_history_store(query.query_sha256),
            history_queries=[query],
        )


def test_history_backed_rejects_evaluation_or_sim_metadata_store(tmp_path: Path) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    query = _history_query(tmp_path)
    evaluation = _history_store(query.query_sha256)
    evaluation["store_purpose"] = "evaluation"
    with pytest.raises(ValueError, match="store_purpose"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=evaluation,
            history_queries=[query],
        )

    sim_metadata = _history_store(query.query_sha256)
    sim_metadata["source_classes"] = ["camera_rgb", "sim_metadata"]
    with pytest.raises(ValueError, match="source_classes"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=sim_metadata,
            history_queries=[query],
        )

    wrong_schema = _history_store(query.query_sha256)
    wrong_schema["store_schema_version"] = 2
    with pytest.raises(ValueError, match="store_schema_version"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=wrong_schema,
            history_queries=[query],
        )


def test_history_backed_requires_trusted_nonempty_store_queries(tmp_path: Path) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    empty = _history_query(tmp_path, nonempty=False)
    with pytest.raises(ValueError, match="nonempty production-history evidence"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=_history_store(empty.query_sha256),
            history_queries=[empty],
        )

    with pytest.raises(TypeError, match="trusted HistoryQuery"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=_history_store("8" * 64),
            history_queries=[{"query_sha256": "8" * 64}],
        )


def test_history_backed_remains_pending_without_separate_route_approval(
    tmp_path: Path,
) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    query = _history_query(tmp_path)
    receipt = preflight_explanation_route(
        requested_mode="history_backed",
        expected_test_vials=18,
        expected_real_check_vials=30,
        predicted_mask_provenance=_mask_provenance(),
        history_store_provenance=_history_store(query.query_sha256),
        history_queries=[query],
    )

    assert receipt["selected_mode"] == "history_backed"
    assert receipt["fallback_used"] is False
    assert receipt["activation_status"] == "pending_inn62_section6_approval"
    assert receipt["history_backed_explanations_enabled"] is False
    assert receipt["route_decision"] is None
    assert receipt["history"]["query_sha256s"] == [query.query_sha256]
    assert receipt["history"]["member_count"] == len(query.members) == 2

    drifted = _history_store(query.query_sha256)
    drifted["query_sha256s"] = ["9" * 64]
    with pytest.raises(ValueError, match="query_sha256s"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=deepcopy(_mask_provenance()),
            history_store_provenance=drifted,
            history_queries=[query],
        )


def test_history_backed_enables_only_with_exact_hash_bound_liam_route_approval(
    tmp_path: Path,
) -> None:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    query = _history_query(tmp_path)
    decision = _route_decision()
    decision_sha256 = _canonical_hash(decision)
    receipt = preflight_explanation_route(
        requested_mode="history_backed",
        expected_test_vials=18,
        expected_real_check_vials=30,
        predicted_mask_provenance=_mask_provenance(),
        history_store_provenance=_history_store(query.query_sha256),
        history_queries=[query],
        route_decision=decision,
        expected_route_decision_sha256=decision_sha256,
    )

    assert receipt["activation_status"] == "history_backed_requirements_satisfied"
    assert receipt["history_backed_explanations_enabled"] is True
    assert receipt["route_decision"] == {
        "issue_id": "INN-62",
        "scope": "section_6_history_backed_explanations",
        "approved_mode": "history_backed",
        "approver_name": "Liam",
        "approver_handle": "@salasl",
        "route_decision_sha256": decision_sha256,
    }

    wrong_hash = "f" * 64
    with pytest.raises(ValueError, match="route-decision SHA-256"):
        preflight_explanation_route(
            requested_mode="history_backed",
            expected_test_vials=18,
            expected_real_check_vials=30,
            predicted_mask_provenance=_mask_provenance(),
            history_store_provenance=_history_store(query.query_sha256),
            history_queries=[query],
            route_decision=decision,
            expected_route_decision_sha256=wrong_hash,
        )


def test_pending_history_preflight_cannot_reach_claim_or_input_read(
    tmp_path: Path, monkeypatch
) -> None:
    import fluidmix.carryover_pipeline as pipeline

    query = _history_query(tmp_path)
    pending = pipeline.preflight_explanation_route(
        requested_mode="history_backed",
        expected_test_vials=18,
        expected_real_check_vials=30,
        predicted_mask_provenance=_mask_provenance(),
        history_store_provenance=_history_store(query.query_sha256),
        history_queries=[query],
    )
    reads: list[Path] = []
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda path, _role: reads.append(Path(path)) or b"unexpected",
    )

    with pytest.raises(ValueError, match="pending approval"):
        pipeline.run_test(
            video=tmp_path / "video.bin",
            recipe_sequence=tmp_path / "recipe.json",
            freeze=tmp_path / "freeze.json",
            expected_freeze_sha256="0" * 64,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
            preflight_receipt=pending,
            expected_preflight_sha256=str(pending["preflight_sha256"]),
            selected_mode="history_backed",
        )
    assert reads == []
    assert not (tmp_path / "claim.json").exists()

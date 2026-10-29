"""CPU-only contracts for selecting the dormant WP7 explanation route."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha(path)


def _vial(
    root: Path, *, vial: int, role: str, start_cursor: int
) -> tuple[list[dict[str, object]], dict[str, dict[str, str]], dict[str, int]]:
    clip = root / "clips" / f"RUN-{vial}.mp4"
    clip_sha = _write(clip, f"synthetic-clip-{vial}".encode())
    records: list[dict[str, object]] = []
    locators: dict[str, dict[str, str]] = {}
    cursors: dict[str, int] = {}
    for stage_ordinal, tank in enumerate(("t100", "t200"), start=1):
        evidence_id = f"E-RUN-{vial}-{stage_ordinal}"
        image_id = f"I-RUN-{vial}-{stage_ordinal}"
        rgb = root / "rgb" / f"{image_id}.png"
        mask = root / "predicted" / f"{image_id}.png"
        rgb_sha = _write(rgb, f"synthetic-rgb-{evidence_id}".encode())
        mask_sha = _write(mask, f"synthetic-prediction-{evidence_id}".encode())
        records.append(
            {
                "run_id": "RUN",
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
                "measurement_id": f"M-{vial}-{stage_ordinal}",
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


def _history_query(tmp_path: Path, *, nonempty: bool = True):
    from fluidmix.evidence_store import EvidenceStore

    artifact_root = tmp_path / ("history-artifacts" if nonempty else "empty-artifacts")
    batches = [
        _vial(artifact_root, vial=1, role="train", start_cursor=10),
        _vial(artifact_root, vial=2, role="train", start_cursor=20),
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
        allowed_roles={"train"},
        target_evidence_id="E-RUN-2-1" if nonempty else "E-RUN-1-1",
    )


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


def test_history_backed_accepts_only_hash_bound_nonempty_production_history(
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
    assert receipt["history_backed_explanations_enabled"] is True
    assert receipt["test"]["terminal_behavior"] == "history_backed_explanation"
    assert receipt["real_check"]["terminal_behavior"] == "history_backed_explanation"
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

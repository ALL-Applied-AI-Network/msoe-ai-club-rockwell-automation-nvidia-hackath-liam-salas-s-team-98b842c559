"""Post-hoc scoring of frozen synthetic vial decisions against separated labels."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))
SCRIPT = ROOT / "cli/evaluation/scoring" / "score_synthetic_stage_decisions.py"

from evaluation.scoring.synthetic_decision_scoring import (
    score_synthetic_stage_decision_files,
    score_synthetic_stage_decisions,
)


MEASURED_SHA = "a" * 64
LABEL_SHA = "b" * 64


def _artifacts() -> tuple[dict[str, object], dict[str, object]]:
    label_profiles = (
        ("REF-A", "reference_only", ("keep", "keep", "keep")),
        ("REF-B", "reference_only", ("keep", "keep", "keep")),
        ("TARGET-CLEAN", "scored_target", ("keep", "keep", "keep")),
        ("TARGET-FIRST", "scored_target", ("discard", "keep", "keep")),
        ("TARGET-TWO", "scored_target", ("discard", "discard", "keep")),
        ("TARGET-ALL", "scored_target", ("discard", "discard", "discard")),
    )
    labels: list[dict[str, object]] = []
    for batch_id, role, dispositions in label_profiles:
        for ordinal, disposition in enumerate(dispositions, start=1):
            residual = 0.0 if disposition == "keep" else 0.5
            labels.append(
                {
                    "case_id": f"CASE-{batch_id}-V{ordinal}",
                    "batch_id": batch_id,
                    "batch_role": role,
                    "vial_ordinal": ordinal,
                    "synthetic_profile_id": batch_id.lower(),
                    "prescribed_leading_residual_volume_ml": residual,
                    "prescribed_leading_residual_particle_count_per_dispense": (
                        0 if residual == 0.0 else 44
                    ),
                    "expected_disposition": disposition,
                    "label_source": "prescribed_synthetic_latent_oracle",
                    "render_schedule_sha256": f"{len(labels) + 1:064x}",
                    "excluded_from_model_inputs": True,
                    "excluded_from_deploy_inputs": True,
                }
            )
    predictions = {
        ("TARGET-CLEAN", 1): "keep",
        ("TARGET-CLEAN", 2): "keep",
        ("TARGET-CLEAN", 3): "insufficient_evidence",
        ("TARGET-FIRST", 1): "discard",
        ("TARGET-FIRST", 2): "keep",
        ("TARGET-FIRST", 3): "discard",
        ("TARGET-TWO", 1): "keep",
        ("TARGET-TWO", 2): "discard",
        ("TARGET-TWO", 3): "keep",
        ("TARGET-ALL", 1): "keep",
        ("TARGET-ALL", 2): "discard",
        ("TARGET-ALL", 3): "insufficient_evidence",
    }
    decisions = [
        {
            "decision_id": f"DEC-{batch_id}-V{ordinal}",
            "family_id": "FAMILY-TRAIN",
            "batch_id": batch_id,
            "cohort": "scored_target",
            "partition": "train",
            "vial_id": f"V{ordinal}",
            "vial_position": ordinal,
            "disposition": disposition,
        }
        for (batch_id, ordinal), disposition in predictions.items()
    ]
    measured = {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "data_origin": "twin",
        "real_data_used": False,
        "source_model_input_sha256": "c" * 64,
        "source_capture_manifest_sha256": [f"{index:064x}" for index in range(1, 7)],
        "decision_records": decisions,
        "denominators": {
            "vial_decisions": 12,
            "valid_vial_decisions": 10,
            "abstained_vial_decisions": 2,
        },
    }
    oracle = {
        "schema_version": 1,
        "artifact_type": "synthetic_train_label_provenance_v2",
        "status": "unrendered",
        "family_id": "FAMILY-TRAIN",
        "vial_labels": labels,
        "denominators": {
            "keep_vials": 12,
            "discard_vials": 6,
            "reference_only_vials": 6,
            "scored_target_vials": 12,
            "scored_target_keep_vials": 6,
            "scored_target_discard_vials": 6,
        },
    }
    return measured, oracle


def test_scores_keep_discard_and_abstention_without_changing_denominators() -> None:
    measured, oracle = _artifacts()

    score = score_synthetic_stage_decisions(
        measured,
        oracle,
        measured_evidence_sha256=MEASURED_SHA,
        label_provenance_sha256=LABEL_SHA,
    )

    assert score["source_sha256"] == {
        "measured_evidence": MEASURED_SHA,
        "label_provenance": LABEL_SHA,
    }
    assert score["denominators"] == {
        "oracle_rows": 18,
        "reference_only_rows": 6,
        "scored_target_rows": 12,
        "expected_keep": 6,
        "expected_discard": 6,
        "decisions": 12,
        "covered_decisions": 10,
        "abstained_decisions": 2,
    }
    assert score["confusion"] == {
        "expected_keep_predicted_keep": 4,
        "expected_keep_predicted_discard": 1,
        "expected_keep_abstained": 1,
        "expected_discard_predicted_keep": 2,
        "expected_discard_predicted_discard": 3,
        "expected_discard_abstained": 1,
    }
    assert score["metrics"] == {
        "coverage": {"numerator": 10, "denominator": 12, "fraction": 10 / 12},
        "exact_match_all": {"numerator": 7, "denominator": 12, "fraction": 7 / 12},
        "covered_accuracy": {"numerator": 7, "denominator": 10, "fraction": 0.7},
        "keep_recall": {"numerator": 4, "denominator": 6, "fraction": 4 / 6},
        "discard_recall": {"numerator": 3, "denominator": 6, "fraction": 0.5},
    }
    assert len(score["case_results"]) == 12
    assert {row["outcome"] for row in score["case_results"]} == {
        "correct",
        "incorrect",
        "abstained",
    }
    assert score["real_data_used"] is False


def _score(measured: object, oracle: object) -> dict[str, object]:
    return score_synthetic_stage_decisions(
        measured,
        oracle,
        measured_evidence_sha256=MEASURED_SHA,
        label_provenance_sha256=LABEL_SHA,
    )


def test_rejects_duplicate_decision_and_case_identities() -> None:
    measured, oracle = _artifacts()
    measured_duplicate = deepcopy(measured)
    measured_duplicate["decision_records"][1] = deepcopy(
        measured_duplicate["decision_records"][0]
    )

    with pytest.raises(ValueError, match="duplicate identities"):
        _score(measured_duplicate, oracle)

    oracle_duplicate = deepcopy(oracle)
    oracle_duplicate["vial_labels"][1]["case_id"] = oracle_duplicate[
        "vial_labels"
    ][0]["case_id"]
    with pytest.raises(ValueError, match="duplicate identities"):
        _score(measured, oracle_duplicate)


def test_rejects_missing_or_extra_scored_case_join() -> None:
    measured, oracle = _artifacts()
    drifted = deepcopy(measured)
    drifted["decision_records"][0]["batch_id"] = "TARGET-EXTRA"

    with pytest.raises(ValueError, match="decision and scored-label IDs differ"):
        _score(drifted, oracle)


def test_rejects_oracle_leakage_and_nonfinite_input() -> None:
    measured, oracle = _artifacts()
    leaked = deepcopy(measured)
    leaked["prescribed_leading_residual_volume_ml"] = 0.5
    with pytest.raises(ValueError, match="forbidden oracle field"):
        _score(leaked, oracle)

    nonfinite = deepcopy(measured)
    nonfinite["unexpected_numeric"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        _score(nonfinite, oracle)


def test_rejects_hash_and_train_family_drift() -> None:
    measured, oracle = _artifacts()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        score_synthetic_stage_decisions(
            measured,
            oracle,
            measured_evidence_sha256="not-a-hash",
            label_provenance_sha256=LABEL_SHA,
        )

    wrong_family = deepcopy(oracle)
    wrong_family["family_id"] = "FAMILY-OTHER"
    with pytest.raises(ValueError, match="measured train family"):
        _score(measured, wrong_family)

    wrong_split = deepcopy(measured)
    wrong_split["decision_records"][0]["partition"] = "test"
    with pytest.raises(ValueError, match="train decisions"):
        _score(wrong_split, oracle)


def test_file_boundary_hash_binds_inputs_and_refuses_clobber(tmp_path: Path) -> None:
    measured, oracle = _artifacts()
    measured_path = tmp_path / "measured.json"
    oracle_path = tmp_path / "labels.json"
    output_path = tmp_path / "score.json"
    for path, value in ((measured_path, measured), (oracle_path, oracle)):
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    measured_hash = hashlib.sha256(measured_path.read_bytes()).hexdigest()
    oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()

    score = score_synthetic_stage_decision_files(
        measured_path,
        oracle_path,
        output_path,
        expected_measured_evidence_sha256=measured_hash,
        expected_label_provenance_sha256=oracle_hash,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == score
    assert score["source_sha256"] == {
        "measured_evidence": measured_hash,
        "label_provenance": oracle_hash,
    }
    with pytest.raises(FileExistsError):
        score_synthetic_stage_decision_files(
            measured_path,
            oracle_path,
            output_path,
            expected_measured_evidence_sha256=measured_hash,
            expected_label_provenance_sha256=oracle_hash,
        )

    other_output = tmp_path / "other-score.json"
    with pytest.raises(ValueError, match="measured evidence SHA-256 mismatch"):
        score_synthetic_stage_decision_files(
            measured_path,
            oracle_path,
            other_output,
            expected_measured_evidence_sha256="d" * 64,
            expected_label_provenance_sha256=oracle_hash,
        )
    assert not other_output.exists()


def test_file_boundary_rejects_bad_inference_hash_before_label_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluation.scoring.synthetic_decision_scoring as scoring

    measured, oracle = _artifacts()
    measured_path = tmp_path / "measured.json"
    label_path = tmp_path / "part-a-labels.json"
    for path, value in ((measured_path, measured), (label_path, oracle)):
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
    opened: list[Path] = []
    real_load = scoring.load_strict_json_snapshot

    def recording_load(path: object, *, max_bytes: int):
        opened.append(Path(path))
        return real_load(Path(path), max_bytes=max_bytes)

    monkeypatch.setattr(scoring, "load_strict_json_snapshot", recording_load)

    with pytest.raises(ValueError, match="measured evidence SHA-256 mismatch"):
        scoring.score_synthetic_stage_decision_files(
            measured_path,
            label_path,
            tmp_path / "score.json",
            expected_measured_evidence_sha256="d" * 64,
            expected_label_provenance_sha256=label_hash,
        )

    assert opened == [measured_path]


def test_cli_scores_only_with_declared_input_hashes(tmp_path: Path) -> None:
    measured, oracle = _artifacts()
    measured_path = tmp_path / "measured.json"
    oracle_path = tmp_path / "labels.json"
    output_path = tmp_path / "score.json"
    for path, value in ((measured_path, measured), (oracle_path, oracle)):
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    measured_hash = hashlib.sha256(measured_path.read_bytes()).hexdigest()
    oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    command = [
        sys.executable,
        str(SCRIPT),
        "--measured-evidence",
        str(measured_path),
        "--measured-evidence-sha256",
        measured_hash,
        "--label-provenance",
        str(oracle_path),
        "--label-provenance-sha256",
        oracle_hash,
        "--out",
        str(output_path),
    ]

    result = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "scored 12/12 synthetic target decisions" in result.stdout
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "artifact_type"
    ] == "synthetic_stage_decision_score"


def test_accepts_current_separated_train_label_artifact() -> None:
    from twin.cohorts.train_residual_manifest_v2 import (
        build_train_residual_artifacts,
    )

    labels = build_train_residual_artifacts(repo_root=ROOT)[
        "label-provenance.json"
    ]
    scored = [
        row for row in labels["vial_labels"] if row["batch_role"] == "scored_target"
    ]
    measured = {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "data_origin": "twin",
        "real_data_used": False,
        "source_model_input_sha256": "c" * 64,
        "source_capture_manifest_sha256": [f"{index:064x}" for index in range(1, 7)],
        "decision_records": [
            {
                "decision_id": f"DEC-{row['batch_id']}-V{row['vial_ordinal']}",
                "family_id": labels["family_id"],
                "batch_id": row["batch_id"],
                "cohort": "scored_target",
                "partition": "train",
                "vial_id": f"V{row['vial_ordinal']}",
                "vial_position": row["vial_ordinal"],
                "disposition": row["expected_disposition"],
            }
            for row in scored
        ],
        "denominators": {
            "vial_decisions": 12,
            "valid_vial_decisions": 12,
            "abstained_vial_decisions": 0,
        },
    }

    score = _score(measured, labels)

    assert score["metrics"]["exact_match_all"] == {
        "numerator": 12,
        "denominator": 12,
        "fraction": 1.0,
    }
    assert {row["case_id"] for row in score["case_results"]} == {
        row["case_id"] for row in scored
    }

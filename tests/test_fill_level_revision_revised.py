"""CPU-only tests for revised-model-only synthetic validation evidence."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_fill_level_revision_revised.py"
SPEC = importlib.util.spec_from_file_location(
    "evaluate_fill_level_revision_revised", SCRIPT
)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def _args(tmp_path):
    return argparse.Namespace(
        dataset_root=tmp_path / "dataset",
        batch_summary=tmp_path / "batch.json",
        exception_review=tmp_path / "review.json",
        revised_training_summary=tmp_path / "training.json",
        revised_training_summary_sha256="a" * 64,
        revised_checkpoint=tmp_path / "best.pt",
        revised_checkpoint_sha256="b" * 64,
        revised_rule_summary=tmp_path / "rule.json",
        revised_rule_summary_sha256="c" * 64,
        output_root=tmp_path / "output",
        device="cpu",
        batch_size=16,
        workers=0,
    )


def test_parser_requires_only_revised_artifact_arguments(tmp_path):
    args = EVALUATOR._parser().parse_args(
        [
            "--dataset-root", str(tmp_path / "dataset"),
            "--batch-summary", str(tmp_path / "batch.json"),
            "--exception-review", str(tmp_path / "review.json"),
            "--revised-training-summary", str(tmp_path / "training.json"),
            "--revised-training-summary-sha256", "a" * 64,
            "--revised-checkpoint", str(tmp_path / "best.pt"),
            "--revised-checkpoint-sha256", "b" * 64,
            "--revised-rule-summary", str(tmp_path / "rule.json"),
            "--revised-rule-summary-sha256", "c" * 64,
            "--output-root", str(tmp_path / "output"),
            "--device", "cpu",
        ]
    )

    assert args.revised_checkpoint == tmp_path / "best.pt"
    assert not hasattr(args, "original_checkpoint")


def test_preflight_binds_one_revised_model_to_combined_audit(tmp_path, monkeypatch):
    args = _args(tmp_path)
    audit = {
        "expected_role_counts": {"train": 3480, "val": 360},
        "eligible_counts": {"train": 3480, "val": 360},
        "batch_summary_sha256": "d" * 64,
        "review_sha256": "e" * 64,
    }
    calls = []

    monkeypatch.setattr(
        EVALUATOR.training,
        "validate_exception_review",
        lambda batch, review: audit,
    )

    def validate_model(*values, **keywords):
        calls.append(("model", keywords["revision"], values))
        return {"checkpoint_sha256": "b" * 64}

    def validate_rule(*values, **keywords):
        calls.append(("rule", keywords["revision"], values))
        return {"summary_sha256": "c" * 64}

    monkeypatch.setattr(
        EVALUATOR.revision_eval, "validate_training_artifact", validate_model
    )
    monkeypatch.setattr(
        EVALUATOR.revision_eval, "validate_rule_artifact", validate_rule
    )

    observed_audit, evidence = EVALUATOR._preflight(args)

    assert observed_audit is audit
    assert evidence == {
        "model": {"checkpoint_sha256": "b" * 64},
        "rule": {"summary_sha256": "c" * 64},
    }
    assert [(kind, revision) for kind, revision, _ in calls] == [
        ("model", "revised"),
        ("rule", "revised"),
    ]


def test_summary_is_revised_only_and_reports_measured_dark_hsv():
    histograms = np.zeros((1, 256), dtype=np.int64)
    histograms[0, 13] = 2
    measured = EVALUATOR.revision_eval.summarize_darkened_validation_hsv(
        histograms,
        histograms,
    )
    audit = {
        "batch_summary_sha256": "d" * 64,
        "review_sha256": "e" * 64,
    }
    metrics = {
        "normal": {"iou": {"macro": 0.8}, "surface": {}},
        "darkened": {"iou": {"macro": 0.7}, "surface": {}},
    }
    hidden_rule_before_after = {
        "evidence_label": (
            "same_revised_model_predictions_rescored_without_reinference"
        ),
        "variants": {
            "normal": {
                "old_step6_validation": {"frame_count": 300},
                "new_exact_v4_validation": {"frame_count": 60},
            }
        },
        "normal_after_hidden_target": {
            "minimum_rate": 0.95,
            "cohorts": {
                "old_step6_validation": {"passed": True},
                "new_exact_v4_validation": {"passed": True},
            },
            "all_cohorts_passed": True,
        },
    }
    summary = EVALUATOR._build_summary(
        audit=audit,
        evidence={
            "model": {
                "checkpoint_sha256": "b" * 64,
                "dataset_population_contract": {
                    "expected_role_counts": {"train": 3480, "val": 360},
                    "eligible_role_counts": {"train": 3480, "val": 360},
                    "role_count_source": "batch_summary.batch_contract",
                },
            },
            "rule": {
                "summary_sha256": "c" * 64,
                "rule_config": {"measurement_rows_full_half_open": [83, 427]},
            },
        },
        samples=[SimpleNamespace(seed=1100000), SimpleNamespace(seed=1101059)],
        preflight={"ordered_hash_ledger_sha256": "f" * 64},
        metrics=metrics,
        hidden_rule_before_after=hidden_rule_before_after,
        darkened_hsv=measured,
        gate_value=2.5,
        gate_error=None,
        device="cpu",
        torch_version="test",
        gpu=None,
        batch_size=16,
        workers=0,
        elapsed_seconds=1.0,
    )

    assert summary["status"] == "complete"
    assert summary["model_revision"] == "revised"
    assert summary["model"]["metrics"] == metrics
    assert "models" not in summary
    assert "comparison" not in summary
    assert summary["hidden_rule_before_after"] == hidden_rule_before_after
    assert summary["gate"]["visible_surface"]["passed"] is True
    assert summary["gate"]["hidden_surface"] == (
        hidden_rule_before_after["normal_after_hidden_target"]
    )
    assert summary["gate"]["passed"] is True
    assert summary["dataset"]["frame_count"] == 2
    assert summary["dataset"]["test_manifests_opened"] == 0
    assert summary["dataset"]["real_frames_loaded"] == 0
    assert summary["darkened_validation"][
        "measured_post_transform_rgb8_hsv"
    ] == measured


def test_run_refuses_reuse_and_records_rule_and_gate_terminal_evidence(
    tmp_path, monkeypatch
):
    class CpuDevice:
        type = "cpu"

        def __str__(self):
            return "cpu"

    fake_torch = SimpleNamespace(
        __version__="test",
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda device: None,
        ),
        device=lambda value: CpuDevice(),
        utils=SimpleNamespace(
            data=SimpleNamespace(DataLoader=lambda *values, **keywords: object())
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    audit = {
        "batch_summary_sha256": "d" * 64,
        "review_sha256": "e" * 64,
    }
    frozen_surface = {
        "by_case": {"all": {"p95_absolute_error_rows": 4.0}},
        "by_level_bin_10ml": {},
    }
    evidence = {
        "model": {"checkpoint_sha256": "b" * 64},
        "rule": {
            "summary_sha256": "c" * 64,
            "ordered_hash_ledger_sha256": "f" * 64,
            "rule_config": {"measurement_rows_full_half_open": [83, 427]},
            "aggregate": frozen_surface,
        },
    }
    samples = [SimpleNamespace(seed=1100000), SimpleNamespace(seed=1101059)]
    measured_histograms = np.zeros((1, 256), dtype=np.int64)
    measured_histograms[0, 13] = 2
    measured = EVALUATOR.revision_eval.summarize_darkened_validation_hsv(
        measured_histograms,
        measured_histograms,
    )
    hidden_rule_before_after = {
        "evidence_label": (
            "same_revised_model_predictions_rescored_without_reinference"
        ),
        "variants": {},
        "normal_after_hidden_target": {
            "minimum_rate": 0.95,
            "cohorts": {
                "old_step6_validation": {
                    "predicted_hidden_rate": 0.96,
                    "true_in_range_over_all_frames_rate": 0.96,
                    "passed": True,
                },
                "new_exact_v4_validation": {
                    "predicted_hidden_rate": 0.97,
                    "true_in_range_over_all_frames_rate": 0.97,
                    "passed": True,
                },
            },
            "all_cohorts_passed": True,
        },
    }
    runtime = {
        "metrics": {
            "normal": {
                "iou": {"macro": 0.8},
                "surface": frozen_surface,
                "per_frame": [],
            },
            "darkened": {
                "iou": {"macro": 0.7},
                "surface": frozen_surface,
                "per_frame": [],
            },
            "hidden_rule_before_after": hidden_rule_before_after,
        }
    }

    monkeypatch.setattr(EVALUATOR, "_preflight", lambda args: (audit, evidence))
    monkeypatch.setattr(EVALUATOR.training, "seed_everything", lambda seed: None)
    monkeypatch.setattr(
        EVALUATOR.surface_eval,
        "discover_validation_frames",
        lambda root, observed_audit: (samples, {sample.seed: {} for sample in samples}),
    )
    monkeypatch.setattr(
        EVALUATOR.training,
        "verify_sample_files",
        lambda observed: {"ordered_hash_ledger_sha256": "f" * 64},
    )
    monkeypatch.setattr(EVALUATOR.shared, "_make_dataset", lambda torch, values: object())
    monkeypatch.setattr(
        EVALUATOR.shared,
        "_load_model",
        lambda torch, checkpoint, model_evidence, device: object(),
    )
    monkeypatch.setattr(
        EVALUATOR.shared,
        "_infer",
        lambda *values, **keywords: ({"revised": runtime["metrics"]}, measured),
    )
    monkeypatch.setattr(
        EVALUATOR.surface_eval, "validate_case_counts", lambda *values: {}
    )

    # Existing evidence roots are never reused or mutated.
    reuse_args = _args(tmp_path / "reuse")
    reuse_args.output_root.mkdir(parents=True)
    marker = reuse_args.output_root / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        EVALUATOR.run(reuse_args)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not (reuse_args.output_root / "state.jsonl").exists()

    # A frozen-rule reproduction mismatch becomes explicit failure evidence.
    mismatch_args = _args(tmp_path / "mismatch")
    mismatch_args.output_root = tmp_path / "mismatch-output"
    runtime["metrics"] = {
        **runtime["metrics"],
        "normal": {
            **runtime["metrics"]["normal"],
            "surface": {"by_case": {}, "by_level_bin_10ml": {}},
        },
    }
    with pytest.raises(ValueError, match="do not reproduce the frozen rule"):
        EVALUATOR.run(mismatch_args)
    failure = json.loads(
        (mismatch_args.output_root / "failure.json").read_text(encoding="utf-8")
    )
    mismatch_states = [
        json.loads(line)["status"]
        for line in (mismatch_args.output_root / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert failure["error_type"] == "ValueError"
    assert mismatch_states == ["RUNNING", "FAILED"]

    # A valid reproduction that misses the p95 gate writes a GATE_FAILED terminal state.
    gate_args = _args(tmp_path / "gate")
    gate_args.output_root = tmp_path / "gate-output"
    runtime["metrics"] = {
        "normal": {
            "iou": {"macro": 0.8},
            "surface": frozen_surface,
            "per_frame": [],
        },
        "darkened": {
            "iou": {"macro": 0.7},
            "surface": frozen_surface,
            "per_frame": [],
        },
        "hidden_rule_before_after": hidden_rule_before_after,
    }
    monkeypatch.setattr(
        EVALUATOR.revision_eval,
        "enforce_normal_validation_gate",
        lambda metrics: (_ for _ in ()).throw(ValueError("p95 exceeds 3 rows")),
    )
    hidden_gate_calls = []
    monkeypatch.setattr(
        EVALUATOR.revision_eval,
        "enforce_hidden_validation_gate",
        lambda comparison: hidden_gate_calls.append(comparison)
        or comparison["normal_after_hidden_target"],
    )
    result = EVALUATOR.run(gate_args)
    full_summary = json.loads(
        (gate_args.output_root / "summary.json").read_text(encoding="utf-8")
    )
    gate_states = [
        json.loads(line)["status"]
        for line in (gate_args.output_root / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["status"] == "gate_failed"
    assert full_summary["gate"]["passed"] is False
    assert full_summary["gate"]["error"] == "p95 exceeds 3 rows"
    assert full_summary["gate"]["hidden_surface"] == (
        hidden_rule_before_after["normal_after_hidden_target"]
    )
    assert full_summary["hidden_rule_before_after"] == hidden_rule_before_after
    assert hidden_gate_calls == [hidden_rule_before_after]
    assert gate_states == ["RUNNING", "GATE_FAILED"]

    # A hidden-target miss also preserves the summary and writes GATE_FAILED.
    hidden_args = _args(tmp_path / "hidden-gate")
    hidden_args.output_root = tmp_path / "hidden-gate-output"
    runtime["metrics"] = {
        "normal": {
            "iou": {"macro": 0.8},
            "surface": frozen_surface,
            "per_frame": [],
        },
        "darkened": {
            "iou": {"macro": 0.7},
            "surface": frozen_surface,
            "per_frame": [],
        },
        "hidden_rule_before_after": hidden_rule_before_after,
    }
    monkeypatch.setattr(
        EVALUATOR.revision_eval,
        "enforce_normal_validation_gate",
        lambda metrics: 2.0,
    )
    monkeypatch.setattr(
        EVALUATOR.revision_eval,
        "enforce_hidden_validation_gate",
        lambda comparison: (_ for _ in ()).throw(
            ValueError(
                "old_step6_validation predicted_hidden_rate=0.940000"
            )
        ),
    )
    result = EVALUATOR.run(hidden_args)
    hidden_summary = json.loads(
        (hidden_args.output_root / "summary.json").read_text(encoding="utf-8")
    )
    hidden_states = [
        json.loads(line)["status"]
        for line in (hidden_args.output_root / "state.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert result["status"] == "gate_failed"
    assert hidden_summary["gate"]["visible_surface"]["passed"] is True
    assert hidden_summary["gate"]["passed"] is False
    assert "predicted_hidden_rate=0.940000" in hidden_summary["gate"]["error"]
    assert hidden_states == ["RUNNING", "GATE_FAILED"]

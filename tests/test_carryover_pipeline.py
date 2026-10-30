"""Synthetic boundary tests for the one-pass production carryover scaffold."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/hackathon5/liamreporoot/shared-ml-env/.venv/bin/python")
sys.path.insert(0, str(ROOT / "src"))


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _validation_results(*, accepts_real_rgb_only: bool = False) -> dict[str, object]:
    from fluidmix.carryover_pipeline import (
        RECIPE_SEQUENCE_SCHEMA_SHA256,
        pipeline_source_sha256,
    )

    decision_config = {
        "discard_threshold_delta_e76": 2.0,
        "keep_threshold_delta_e76": 1.0,
    }
    return {
        "schema_version": "carryover-validation-results-v1",
        "artifact_type": "carryover_validation_results",
        "split_role": "validation",
        "pipeline_source_commit": "e4d93e4c24c274ec3e99df220204e5f86443f777",
        "pipeline_source_sha256": pipeline_source_sha256(),
        "recipe_schema_sha256": RECIPE_SEQUENCE_SCHEMA_SHA256,
        "mask_runtime": {
            "runtime_id": "reviewed-clamp-runtime-v1",
            "runtime_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "accepts_real_rgb_only": accepts_real_rgb_only,
        },
        "measurement": {
            "contract_id": "vision-measurement-v1",
            "contract_sha256": "4" * 64,
        },
        "decision": {
            "config": decision_config,
            "config_sha256": _canonical_hash(decision_config),
            "dp5_choice": "baseline",
            "dp5_passed_without_more_errors": False,
        },
        "explanation": {
            "method": "deterministic_guarded_template",
            "method_sha256": "5" * 64,
            "model_id": "none",
            "model_sha256": "6" * 64,
            "prompt_sha256": "7" * 64,
            "guard_sha256": "8" * 64,
        },
        "store": {"schema_version": 3, "schema_sha256": "9" * 64},
        "validation_denominators": {
            "run_count": 2,
            "vial_count": 12,
            "predicted_stage_count": 30,
            "measured_stage_count": 30,
            "history_supported_vial_count": 0,
            "explained_vial_count": 12,
        },
    }


def _recipe(*, split_role: str = "test", vial_count: int = 2) -> dict[str, object]:
    vials = []
    for ordinal in range(1, vial_count + 1):
        vials.append(
            {
                "case_id": f"CASE-{ordinal}",
                "vial_ordinal": ordinal,
                "block_recipe_id": "R-CURRENT",
                "block_position": ordinal,
                "recipe_id": "R-CURRENT",
                "previous_recipe_id": "R-PREVIOUS" if ordinal == 1 else "R-CURRENT",
                "programmed_tank_ml": {"t100": 10.0, "t200": 20.0, "t400": 0.0},
                "dispense_order": ["t100", "t200"],
            }
        )
    return {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": split_role,
        "split_version": "whole-run-v1",
        "runs": [{"run_id": "RUN-ONE", "vials": vials}],
    }


def _freeze(tmp_path: Path, *, accepts_real_rgb_only: bool = False) -> tuple[Path, str]:
    from fluidmix.carryover_pipeline import freeze_validation

    validation = tmp_path / "validation.json"
    output = tmp_path / "freeze.json"
    _write_json(validation, _validation_results(accepts_real_rgb_only=accepts_real_rgb_only))
    receipt = freeze_validation(validation, output)
    return output, str(receipt["freeze_sha256"])


def _evaluation_preflight() -> tuple[dict[str, object], str]:
    from fluidmix.carryover_pipeline import preflight_explanation_route

    receipt = preflight_explanation_route(
        requested_mode="evaluation_only",
        expected_test_vials=18,
        expected_real_check_vials=30,
        evaluation_store_provenance={
            "schema_version": "wp7-evaluation-only-store-v1",
            "artifact_type": "evaluation_only_store_provenance",
            "store_purpose": "evaluation_only",
            "store_schema_version": 3,
            "store_schema_sha256": "a" * 64,
            "store_snapshot_sha256": "b" * 64,
            "review_receipt_sha256": "c" * 64,
            "stage_row_count": 0,
        },
    )
    return receipt, str(receipt["preflight_sha256"])


def test_freeze_validation_is_canonical_strict_and_no_clobber(tmp_path):
    from fluidmix.carryover_pipeline import freeze_validation

    source = tmp_path / "validation.json"
    output = tmp_path / "freeze.json"
    _write_json(source, _validation_results())

    receipt = freeze_validation(source, output)

    assert receipt["schema_version"] == "carryover-freeze-v1"
    assert receipt["validation_results_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["freeze_sha256"] == _canonical_hash(
        {key: value for key, value in receipt.items() if key != "freeze_sha256"}
    )
    with pytest.raises(FileExistsError):
        freeze_validation(source, output)

    invalid = _validation_results()
    invalid["test_path"] = "held-out.mp4"
    _write_json(tmp_path / "invalid.json", invalid)
    with pytest.raises(ValueError, match="fields"):
        freeze_validation(tmp_path / "invalid.json", tmp_path / "never.json")

    wrong_schema = _validation_results()
    wrong_schema["recipe_schema_sha256"] = "0" * 64
    _write_json(tmp_path / "wrong-schema.json", wrong_schema)
    with pytest.raises(ValueError, match="recipe schema SHA-256"):
        freeze_validation(tmp_path / "wrong-schema.json", tmp_path / "never-2.json")


def test_freeze_validation_requires_official_deterministic_dp4_default(tmp_path):
    from fluidmix.carryover_pipeline import freeze_validation

    model_selected = _validation_results()
    model_selected["explanation"]["method"] = "vision_language_model"
    model_selected["explanation"]["model_id"] = "candidate-model"
    source = tmp_path / "model-selected.json"
    _write_json(source, model_selected)

    with pytest.raises(ValueError, match="deterministic DP4/default"):
        freeze_validation(source, tmp_path / "must-not-freeze.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("label", "discard"),
        ("residue_ml", 4.0),
        ("tank_colour_preset", "blue"),
        ("simulation_time", 12.0),
        ("mask_path", "masks/answer.png"),
    ],
)
def test_recipe_parser_accepts_only_programmed_sequence_fields(tmp_path, field, value):
    from fluidmix.carryover_pipeline import load_recipe_sequence

    recipe = _recipe()
    recipe["runs"][0]["vials"][0][field] = value
    path = tmp_path / "recipe.json"
    _write_json(path, recipe)

    with pytest.raises(ValueError, match="forbidden|fields"):
        load_recipe_sequence(path, expected_role="test")


def test_recipe_parser_rejects_forbidden_vocabulary_in_values(tmp_path):
    from fluidmix.carryover_pipeline import load_recipe_sequence

    recipe = _recipe()
    recipe["runs"][0]["vials"][0]["recipe_id"] = "synthetic-oracle-answer"
    path = tmp_path / "recipe.json"
    _write_json(path, recipe)

    with pytest.raises(ValueError, match="forbidden recipe sequence value"):
        load_recipe_sequence(path, expected_role="test")


def test_run_test_verifies_freeze_then_claim_before_any_production_input_read(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    freeze, digest = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    video = tmp_path / "video.bin"
    recipe = tmp_path / "recipe.json"
    video.write_bytes(b"synthetic RGB fixture bytes")
    _write_json(recipe, _recipe())
    reads: list[Path] = []
    original = pipeline._read_production_bytes

    def tracked(path: Path, role: str) -> bytes:
        reads.append(Path(path))
        return original(path, role)

    monkeypatch.setattr(pipeline, "_read_production_bytes", tracked)
    with pytest.raises(ValueError, match="freeze SHA-256"):
        pipeline.run_test(
            video=video,
            recipe_sequence=recipe,
            freeze=freeze,
            expected_freeze_sha256="0" * 64,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
            preflight_receipt=preflight,
            expected_preflight_sha256=preflight_sha256,
            selected_mode="evaluation_only",
        )
    assert reads == []
    assert not (tmp_path / "claim.json").exists()

    bad_recipe = tmp_path / "bad-recipe.json"
    bad_recipe.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="recipe sequence"):
        pipeline.run_test(
            video=video,
            recipe_sequence=bad_recipe,
            freeze=freeze,
            expected_freeze_sha256=digest,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "fatal.json",
            claim=tmp_path / "fatal-claim.json",
            owner_id="test-owner",
            preflight_receipt=preflight,
            expected_preflight_sha256=preflight_sha256,
            selected_mode="evaluation_only",
        )
    assert (tmp_path / "fatal-claim.json").exists()
    assert reads == [bad_recipe.resolve()]
    assert video.resolve() not in reads
    fatal = json.loads((tmp_path / "fatal.json").read_text())
    assert fatal["terminal_status"] == "fatal"
    assert fatal["rerun_permitted"] is False
    assert fatal["denominators"]["count_equation_status"] == (
        "unavailable_before_recipe_validation"
    )


def test_run_test_requires_preflight_before_claim_or_production_input_read(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    freeze, digest = _freeze(tmp_path)
    video = tmp_path / "video.bin"
    recipe = tmp_path / "recipe.json"
    video.write_bytes(b"unread synthetic video bytes")
    _write_json(recipe, _recipe(vial_count=18))
    reads: list[Path] = []
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda path, _role: reads.append(Path(path)) or b"unexpected",
    )

    with pytest.raises(TypeError, match="preflight"):
        pipeline.run_test(
            video=video,
            recipe_sequence=recipe,
            freeze=freeze,
            expected_freeze_sha256=digest,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
        )
    assert reads == []
    assert not (tmp_path / "claim.json").exists()


@pytest.mark.parametrize(
    ("digest_override", "mode", "message"),
    [
        ("0" * 64, "evaluation_only", "preflight SHA-256"),
        (None, "history_backed", "selected mode"),
    ],
)
def test_run_test_rejects_mismatched_preflight_before_claim_or_input_read(
    tmp_path, monkeypatch, digest_override, mode, message
):
    import fluidmix.carryover_pipeline as pipeline

    freeze, freeze_sha256 = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    reads: list[Path] = []
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda path, _role: reads.append(Path(path)) or b"unexpected",
    )

    with pytest.raises(ValueError, match=message):
        pipeline.run_test(
            video=tmp_path / "video.bin",
            recipe_sequence=tmp_path / "recipe.json",
            freeze=freeze,
            expected_freeze_sha256=freeze_sha256,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
            preflight_receipt=preflight,
            expected_preflight_sha256=digest_override or preflight_sha256,
            selected_mode=mode,
        )
    assert reads == []
    assert not (tmp_path / "claim.json").exists()


def test_evaluation_preflight_denominator_mismatch_stops_before_video_read(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    freeze, freeze_sha256 = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    recipe = tmp_path / "recipe.json"
    _write_json(recipe, _recipe(vial_count=2))
    reads: list[Path] = []
    original = pipeline._read_production_bytes

    def tracked(path: Path, role: str) -> bytes:
        reads.append(Path(path))
        return original(path, role)

    monkeypatch.setattr(pipeline, "_read_production_bytes", tracked)
    with pytest.raises(ValueError, match="denominator.*preflight"):
        pipeline.run_test(
            video=tmp_path / "unread-video.bin",
            recipe_sequence=recipe,
            freeze=freeze,
            expected_freeze_sha256=freeze_sha256,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "out.json",
            claim=tmp_path / "claim.json",
            owner_id="test-owner",
            preflight_receipt=preflight,
            expected_preflight_sha256=preflight_sha256,
            selected_mode="evaluation_only",
        )
    assert reads == [recipe.resolve()]
    assert (tmp_path / "claim.json").exists()
    fatal = json.loads((tmp_path / "out.json").read_text())
    assert fatal["terminal_status"] == "fatal"
    assert fatal["preflight_sha256"] == preflight_sha256


def test_run_test_missing_capabilities_abstains_with_exact_denominators_and_no_retry(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    freeze, digest = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    video = tmp_path / "video.bin"
    recipe = tmp_path / "recipe.json"
    video.write_bytes(b"synthetic RGB fixture bytes")
    _write_json(recipe, _recipe(vial_count=18))
    kwargs = {
        "video": video,
        "recipe_sequence": recipe,
        "freeze": freeze,
        "expected_freeze_sha256": digest,
        "store": tmp_path / "store.sqlite",
        "out": tmp_path / "out.json",
        "claim": tmp_path / "claim.json",
        "owner_id": "test-owner",
        "preflight_receipt": preflight,
        "expected_preflight_sha256": preflight_sha256,
        "selected_mode": "evaluation_only",
    }

    receipt = pipeline.run_test(**kwargs)

    assert receipt["terminal_status"] == "complete"
    assert receipt["source_bindings"]["video_sha256"] == hashlib.sha256(video.read_bytes()).hexdigest()
    assert len(receipt["operator_records"]) == 18
    assert all(row["disposition"] == "insufficient_evidence" for row in receipt["operator_records"])
    assert all(
        row["explanation"]["method"] == "deterministic_guarded_template"
        for row in receipt["operator_records"]
    )
    assert all(
        row["reason_codes"]
        == [
            "history_support_unavailable",
            "measurement_unavailable",
            "predicted_mask_runtime_unavailable",
        ]
        for row in receipt["operator_records"]
    )
    assert receipt["explanation_policy"] == {
        "schema_version": "wp7-explanation-policy-v1",
        "dp4_status": "official_default",
        "explanation_method": "deterministic_guarded_template",
        "limited_result": {
            "scope": "pre_access_all_abstention_diagnostic_only",
            "attempt_count_per_method": 16,
            "ordered_shuffled_pair_count": 8,
            "keep_discard_accuracy": None,
            "winner": None,
            "provenance": {
                "protocol_commit": "0ac00ad7906634caf08a51f088343c72c433d1c4",
                "protocol_sha256": "7c8325f40541ceae972aeb438e0d0ff9b6abc6df6a50bae50b815e975e1f8eb4",
                "plan_file_sha256": "ac0d8f1f779867cdb3fd77f12ccfa7d7d0c3baf48ecb2afcc35913d5cf5b19d2",
                "plan_canonical_sha256": "7e50db2c0c5d769664c032930b73f8472275e0296413dfba7c5dfa599799bf3d",
            },
            "methods": [
                {
                    "method": "deterministic_template",
                    "strict_parse_count": 16,
                    "correct_abstention_count": 16,
                    "unsupported_cause_refusal_count": 16,
                    "parsed_pair_count": 8,
                    "abstention_flip_count": 0,
                    "refusal_flip_count": 0,
                    "raw_sha256": "6824c8ee227ca21263283439301d18ece7f119363e7515f34621367abd2d97a8",
                    "score_sha256": "ce036c2af554c201d45f2a600bdde6e301f9566c76e7232324f04553722c5843",
                },
                {
                    "method": "Qwen3-VL-8B-Instruct",
                    "strict_parse_count": 16,
                    "correct_abstention_count": 16,
                    "unsupported_cause_refusal_count": 16,
                    "parsed_pair_count": 8,
                    "abstention_flip_count": 0,
                    "refusal_flip_count": 0,
                    "median_generation_seconds": 5.589,
                    "peak_cuda_allocated_bytes": 18824709120,
                    "raw_sha256": "a61c7f5aea4f1b95e67f161c65574b25a0643194345f238045f5ba0cd7c713d7",
                    "score_sha256": "3cd9ccd37af3453063f4aed803bc0c35ee3e2db5832ad10ef6c107a6df9040f9",
                },
                {
                    "method": "AnomalyThink-Qwen3-VL-8B-KCR",
                    "strict_parse_count": 0,
                    "parsed_pair_count": 0,
                    "failure_mode": "free_form_think_and_yes_no",
                    "median_generation_seconds": 15.214,
                    "raw_sha256": "82247421129d1901d7a1b1b0f9b11e1cac99aec14bf074fbd370a4da80e49e24",
                    "score_sha256": "a0dfd8aa795f1c6e3e6aad4b7b5b3c3a12f8ec956a1e976f8fdec31bbea52be3",
                },
                {
                    "method": "Cosmos-Reason2-8B",
                    "strict_parse_count": 0,
                    "parsed_pair_count": 0,
                    "failure_mode": "json_in_markdown_fence_no_repair",
                    "median_generation_seconds": 7.619,
                    "raw_sha256": "0d27529df25105ddf2ec193234425301750b1f07a52da79b977c02740b3d448f",
                    "score_sha256": "3b986e50d1ec0a7140c9ecff54fee5f94b256234a715876e225625e8b70fd2d9",
                },
            ],
        },
        "full_comparison": {"issue_id": "INN-67", "timing": "Friday"},
    }
    assert receipt["denominators"] == {
        "run_count": 1,
        "vial_count": 18,
        "expected_stage_count": 36,
        "terminal_record_count": 18,
        "decision_count": 0,
        "abstention_count": 18,
        "skip_count": 0,
        "coverage_numerator": 0,
        "coverage_denominator": 18,
        "correct_decision_numerator": None,
        "correct_decision_denominator": None,
        "discard_recall_numerator": None,
        "discard_recall_denominator": None,
        "false_discard_count": None,
        "false_keep_count": None,
    }
    assert (tmp_path / "out.json.journal.jsonl").read_text().count("\n") == 3
    assert not (tmp_path / "store.sqlite").exists()
    assert receipt["preflight_binding"] == {
        "selected_mode": "evaluation_only",
        "preflight_sha256": preflight_sha256,
    }
    assert receipt["source_bindings"]["preflight_sha256"] == preflight_sha256
    assert receipt["source_bindings"]["selected_mode"] == "evaluation_only"
    claim = json.loads((tmp_path / "claim.json").read_text())
    assert claim["preflight_sha256"] == preflight_sha256
    assert claim["selected_mode"] == "evaluation_only"
    journal = [
        json.loads(line)
        for line in (tmp_path / "out.json.journal.jsonl").read_text().splitlines()
    ]
    assert all(event["preflight_sha256"] == preflight_sha256 for event in journal)
    assert all(event["selected_mode"] == "evaluation_only" for event in journal)

    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda *_: (_ for _ in ()).throw(AssertionError("rerun read production input")),
    )
    with pytest.raises(FileExistsError):
        pipeline.run_test(**kwargs)


def test_bound_runtime_analyze_projection_and_ingest_cross_one_pass_seam(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    freeze, digest = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    video = tmp_path / "camera-video.bin"
    recipe = tmp_path / "recipe.json"
    video.write_bytes(b"synthetic camera fixture bytes")
    _write_json(recipe, _recipe(vial_count=18))
    marker = pipeline.RuntimeProjectionInput((), object(), {}, {})
    calls = []

    class BoundRuntime:
        runtime_id = "reviewed-clamp-runtime-v1"
        runtime_sha256 = "2" * 64
        config_sha256 = "3" * 64
        accepts_real_rgb_only = True

        def analyze(self, *, video_bytes, recipe_sequence, artifact_root):
            calls.append(("analyze", video_bytes, len(recipe_sequence["runs"]), artifact_root))
            return marker

    monkeypatch.setattr(
        pipeline,
        "project_runtime_measurements",
        lambda value: calls.append(("project", value)) or [{"evidence_id": "E-1"}],
    )
    monkeypatch.setattr(
        pipeline,
        "append_runtime_measurements",
        lambda **kwargs: calls.append(("ingest", kwargs)),
    )

    receipt = pipeline.run_test(
        video=video,
        recipe_sequence=recipe,
        freeze=freeze,
        expected_freeze_sha256=digest,
        store=tmp_path / "store.sqlite",
        out=tmp_path / "out.json",
        claim=tmp_path / "claim.json",
        owner_id="test-owner",
        preflight_receipt=preflight,
        expected_preflight_sha256=preflight_sha256,
        selected_mode="evaluation_only",
        mask_runtime=BoundRuntime(),
    )

    assert [call[0] for call in calls] == ["analyze", "project", "ingest"]
    assert receipt["provenance_inventory"]["measured_stage_count"] == 1
    assert receipt["provenance_inventory"]["model_loaded"] is True
    assert all(
        row["reason_codes"] == ["history_support_unavailable"]
        for row in receipt["operator_records"]
    )
    assert "predicted_mask_runtime_integration_pending" not in json.dumps(receipt)


def test_existing_claim_blocks_input_reads_even_without_an_output(tmp_path, monkeypatch):
    import fluidmix.carryover_pipeline as pipeline

    freeze, digest = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    video = tmp_path / "video.bin"
    recipe = tmp_path / "recipe.json"
    video.write_bytes(b"unread synthetic video bytes")
    _write_json(recipe, _recipe())
    claim = tmp_path / "claim.json"
    claim.write_text("existing one-pass sentinel", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda *_: (_ for _ in ()).throw(AssertionError("claim collision read input")),
    )

    with pytest.raises(FileExistsError):
        pipeline.run_test(
            video=video,
            recipe_sequence=recipe,
            freeze=freeze,
            expected_freeze_sha256=digest,
            store=tmp_path / "store.sqlite",
            out=tmp_path / "out.json",
            claim=claim,
            owner_id="test-owner",
            preflight_receipt=preflight,
            expected_preflight_sha256=preflight_sha256,
            selected_mode="evaluation_only",
        )
    assert not (tmp_path / "out.json").exists()


def test_two_run_bundle_has_exact_run_vial_and_stage_denominators(tmp_path):
    from fluidmix.carryover_pipeline import run_test

    freeze, digest = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    video = tmp_path / "video.bin"
    recipe_path = tmp_path / "recipe.json"
    video.write_bytes(b"two-run synthetic RGB fixture")
    recipe = _recipe(vial_count=9)
    second = json.loads(json.dumps(recipe["runs"][0]))
    second["run_id"] = "RUN-TWO"
    recipe["runs"].append(second)
    _write_json(recipe_path, recipe)

    receipt = run_test(
        video=video,
        recipe_sequence=recipe_path,
        freeze=freeze,
        expected_freeze_sha256=digest,
        store=tmp_path / "store.sqlite",
        out=tmp_path / "out.json",
        claim=tmp_path / "claim.json",
        owner_id="test-owner",
        preflight_receipt=preflight,
        expected_preflight_sha256=preflight_sha256,
        selected_mode="evaluation_only",
    )

    assert receipt["denominators"]["run_count"] == 2
    assert receipt["denominators"]["vial_count"] == 18
    assert receipt["denominators"]["expected_stage_count"] == 36
    assert receipt["denominators"]["terminal_record_count"] == 18
    assert receipt["denominators"]["abstention_count"] == 18


def test_operational_path_alias_is_rejected_before_claim_or_input_read(
    tmp_path, monkeypatch
):
    import fluidmix.carryover_pipeline as pipeline

    freeze, digest = _freeze(tmp_path)
    preflight, preflight_sha256 = _evaluation_preflight()
    video = tmp_path / "video.bin"
    recipe = tmp_path / "recipe.json"
    video.write_bytes(b"unread synthetic video bytes")
    _write_json(recipe, _recipe())
    output_and_claim = tmp_path / "aliased.json"
    monkeypatch.setattr(
        pipeline,
        "_read_production_bytes",
        lambda *_: (_ for _ in ()).throw(AssertionError("path alias read input")),
    )

    with pytest.raises(ValueError, match="paths must be distinct"):
        pipeline.run_test(
            video=video,
            recipe_sequence=recipe,
            freeze=freeze,
            expected_freeze_sha256=digest,
            store=tmp_path / "store.sqlite",
            out=output_and_claim,
            claim=output_and_claim,
            owner_id="test-owner",
            preflight_receipt=preflight,
            expected_preflight_sha256=preflight_sha256,
            selected_mode="evaluation_only",
        )
    assert not output_and_claim.exists()


def test_run_real_check_seals_denominator_bound_rgb_route_skip(tmp_path):
    from fluidmix.carryover_pipeline import run_real_check

    freeze, digest = _freeze(tmp_path, accepts_real_rgb_only=False)
    preflight, preflight_sha256 = _evaluation_preflight()
    video = tmp_path / "real-video.bin"
    recipe = tmp_path / "real-recipe.json"
    video.write_bytes(b"synthetic real-role fixture bytes")
    _write_json(recipe, _recipe(split_role="real_check", vial_count=30))

    receipt = run_real_check(
        video=video,
        recipe_sequence=recipe,
        freeze=freeze,
        expected_freeze_sha256=digest,
        store=tmp_path / "real-store.sqlite",
        out=tmp_path / "real-out.json",
        claim=tmp_path / "real-claim.json",
        owner_id="real-check-owner",
        preflight_receipt=preflight,
        expected_preflight_sha256=preflight_sha256,
        selected_mode="evaluation_only",
    )

    assert receipt["terminal_status"] == "skipped"
    assert receipt["skip_reason"] == "accepted_rgb_only_route_unavailable"
    assert receipt["denominators"]["vial_count"] == 30
    assert receipt["denominators"]["skip_count"] == 30
    assert receipt["denominators"]["terminal_record_count"] == 30
    assert all(row["disposition"] == "insufficient_evidence" for row in receipt["operator_records"])


def test_cli_has_three_phases_and_no_answer_side_argument():
    result = subprocess.run(
        [str(PYTHON), "scripts/run_carryover_pipeline.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert all(
        phase in result.stdout
        for phase in ("freeze-validation", "run-test", "run-real-check")
    )
    assert "--oracle" not in result.stdout

    run_help = subprocess.run(
        [str(PYTHON), "scripts/run_carryover_pipeline.py", "run-test", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run_help.returncode == 0, run_help.stderr
    for forbidden in ("label", "mask", "residue", "palette", "simulation"):
        assert f"--{forbidden}" not in run_help.stdout
    for required in ("--preflight", "--preflight-sha256", "--selected-mode"):
        assert required in run_help.stdout


@pytest.mark.parametrize(
    "alias_target",
    ("video", "recipe_sequence", "freeze", "store", "out", "claim", "journal"),
)
def test_cli_rejects_preflight_path_alias_before_loader_or_outputs(
    tmp_path, monkeypatch, alias_target
):
    spec = importlib.util.spec_from_file_location(
        "run_carryover_pipeline_alias_test",
        ROOT / "scripts/run_carryover_pipeline.py",
    )
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    paths = {
        "video": tmp_path / "video.bin",
        "recipe_sequence": tmp_path / "recipe.json",
        "freeze": tmp_path / "freeze.json",
        "store": tmp_path / "store.sqlite",
        "out": tmp_path / "out.json",
        "claim": tmp_path / "claim.json",
    }
    journal = Path(f"{paths['out']}.journal.jsonl")
    preflight = journal if alias_target == "journal" else paths[alias_target]
    loader_calls = []

    def fail_if_loaded(path):
        loader_calls.append(path)
        raise AssertionError("preflight loader must not run for an aliased path")

    monkeypatch.setattr(cli, "load_preflight_receipt", fail_if_loaded)

    with pytest.raises(ValueError, match="preflight path must not alias"):
        cli.main(
            [
                "run-test",
                "--video",
                str(paths["video"]),
                "--recipe-sequence",
                str(paths["recipe_sequence"]),
                "--freeze",
                str(paths["freeze"]),
                "--freeze-sha256",
                "0" * 64,
                "--store",
                str(paths["store"]),
                "--out",
                str(paths["out"]),
                "--claim",
                str(paths["claim"]),
                "--owner-id",
                "synthetic-owner",
                "--preflight",
                str(preflight),
                "--preflight-sha256",
                "1" * 64,
                "--selected-mode",
                "evaluation_only",
            ]
        )

    assert loader_calls == []
    assert not paths["claim"].exists()
    assert not paths["out"].exists()
    assert not journal.exists()

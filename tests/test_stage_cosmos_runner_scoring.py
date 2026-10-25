"""Tests for the frozen 48-row stage/vial Cosmos executor and scorer."""

from __future__ import annotations

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _stage(decision_id: str, ordinal: int, *, outlier: bool) -> dict:
    target = f"IMG-{decision_id}-S{ordinal}-TARGET"
    representative = f"IMG-{decision_id}-S{ordinal}-REFERENCE"
    delta = 4.25 if outlier else 0.75
    return {
        "evidence_id": f"EVID-{decision_id}-S{ordinal}",
        "measurement_id": f"MEAS-{decision_id}-S{ordinal}",
        "stage_id": f"after_t{ordinal}00",
        "stage_ordinal": ordinal,
        "time_code": (171, 342, 686)[ordinal - 1],
        "target_image_id": target,
        "reference_cases": [
            {
                "mode": "all_reference_batches",
                "disposition_source": True,
                "validity": "valid",
                "limit_codes": [],
                "representative_peer_id": representative,
                "peer_count": 2,
                "peer_batch_count": 2,
                "median_radius_delta_e76_raw": 0.4,
                "max_radius_delta_e76_raw": 0.5,
                "mad_radius_delta_e76_raw": 0.1,
                "pairwise_diameter_delta_e76_raw": 0.8,
                "unit": "delta_e76",
            }
        ],
        "primary_mode": "all_reference_batches",
        "validity": "valid",
        "limit_codes": [],
        "target_delta_e76_raw": delta,
        "target_delta_e76_6dp": f"{delta:.6f}",
        "threshold_delta_e76_raw": 2.0,
        "threshold_delta_e76_6dp": "2.000000",
        "stage_outlier": outlier,
        "unit": "delta_e76",
    }


def _decision(index: int) -> dict:
    disposition = ("discard", "keep", "insufficient_evidence")[index % 3]
    decision_id = f"DEC-{index:02d}"
    stages = [
        _stage(decision_id, ordinal, outlier=disposition == "discard" and ordinal == 2)
        for ordinal in range(1, 4)
    ]
    if disposition == "insufficient_evidence":
        stage = stages[1]
        primary = stage["reference_cases"][0]
        stage.update(
            validity="insufficient_evidence",
            limit_codes=["reference_not_unique_or_too_small"],
            target_delta_e76_raw=None,
            target_delta_e76_6dp=None,
            threshold_delta_e76_raw=None,
            threshold_delta_e76_6dp=None,
            stage_outlier=None,
        )
        primary.update(
            validity="insufficient_evidence",
            limit_codes=["reference_not_unique_or_too_small"],
            representative_peer_id=None,
            peer_count=None,
            peer_batch_count=None,
            median_radius_delta_e76_raw=None,
            max_radius_delta_e76_raw=None,
            mad_radius_delta_e76_raw=None,
            pairwise_diameter_delta_e76_raw=None,
        )
    status = {
        "discard": "DIFFERENCE",
        "keep": "NO_DIFFERENCE",
        "insufficient_evidence": "UNKNOWN",
    }[disposition]
    return {
        "decision_id": decision_id,
        "family_id": "VA",
        "batch_id": f"BATCH-{index:02d}",
        "cohort": "scored_target",
        "partition": "validation",
        "previous_sim_recipe_id": "S0160",
        "current_sim_recipe_id": "S0380",
        "previous_programmed_ml": {"t100": 30, "t200": 20, "t400": 10},
        "current_programmed_ml": {"t100": 10, "t200": 30, "t400": 20},
        "programmed_dispense_order": ["t200", "t400", "t100"],
        "vial_id": f"V{index % 3 + 1}",
        "vial_position": index % 3 + 1,
        "decision_timing": "retrospective_after_complete_three_vial_batch_no_latency_claim",
        "stages": stages,
        "disposition": disposition,
        "explanation_status": status,
        "disposition_source": "deterministic_pipeline_primary_reference_evidence",
        "limit_codes": (
            ["reference_not_unique_or_too_small"]
            if disposition == "insufficient_evidence"
            else []
        ),
    }


def _measured() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "config_id": "three-vial-separated-reference-bank-v1",
        "decision_records": [_decision(index) for index in range(12)],
        "data_origin": "twin",
        "real_data_used": False,
    }


def _completion(raw: str) -> dict:
    return {
        "response_raw": raw,
        "response_utf8_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "generated_suffix_token_ids": [1, 2],
        "generated_suffix_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
        "generated_token_count": 2,
        "terminal_token_id": 2,
        "terminal_token_is_eos": True,
        "length_cap_reached": False,
        "truncated_without_eos": False,
    }


class _FakeBatch(dict):
    def to(self, *args, **kwargs):
        return self


class _FakeTokenizer:
    pass


class _FakeImageProcessor:
    def to_dict(self):
        return {"size": {"height": 480, "width": 640}, "rescale": True}


class _FakeProcessor:
    def __init__(self):
        self.chat_template = "native-template"
        self.tokenizer = _FakeTokenizer()
        self.image_processor = _FakeImageProcessor()
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if kwargs["tokenize"] is False:
            return "rendered-user-message<|im_start|>assistant\n"
        image_count = sum(
            item["type"] == "image" for item in messages[0]["content"]
        )
        result = _FakeBatch(
            input_ids=torch.tensor([[11, 12, 13]], dtype=torch.int64),
            attention_mask=torch.tensor([[1, 1, 1]], dtype=torch.int64),
        )
        if image_count:
            result["pixel_values"] = torch.full(
                (image_count, 2), 0.25, dtype=torch.float32
            )
            result["image_grid_thw"] = torch.tensor(
                [[1, 2, 3]] * image_count, dtype=torch.int64
            )
        return result

    def batch_decode(self, value, **kwargs):
        return ["raw stage completion"]


def _input_config(processor) -> dict:
    canonical = json.dumps(
        processor.image_processor.to_dict(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return {
        "experiment_id": "stage-cosmos-nano-va-v1",
        "model_revision": "e59a53c25979a090fa8706c9acc0c254a6e89b92",
        "model_view_manifest_sha256": "a" * 64,
        "prompt_template_sha256": hashlib.sha256(
            (ROOT / "data/cosmos_stage_prompt_v1.txt").read_bytes()
        ).hexdigest(),
        "runtime_versions": {"transformers": "5.17.0"},
        "expected_processor_class": type(processor).__name__,
        "expected_tokenizer_class": type(processor.tokenizer).__name__,
        "chat_template_text_sha256": hashlib.sha256(
            processor.chat_template.encode()
        ).hexdigest(),
        "image_preprocessor_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "chat_template_generation_suffix": "<|im_start|>assistant\n",
        "max_images_per_attempt": 6,
        "image_width": 640,
        "image_height": 480,
        "do_sample": False,
        "max_new_tokens": 384,
        "native_eos_token_ids": [151643, 151645],
    }


def _locators(tmp_path: Path, plan: list[dict]) -> dict[str, dict[str, str]]:
    image_ids = sorted(
        {
            image_id
            for spec in plan
            if spec["kind"] == "model"
            for image_id in spec["image_ids"]
        }
    )
    result = {}
    for index, image_id in enumerate(image_ids):
        path = tmp_path / f"{index:03d}.png"
        Image.new("RGB", (640, 480), (index % 255, 10, 20)).save(path)
        result[image_id] = {
            "rgb_path": str(path),
            "rgb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "mask_path": str(tmp_path / f"mask-{index:03d}.png"),
            "mask_sha256": "b" * 64,
        }
    return result


def test_multi_image_processor_audit_covers_all_36_model_inputs_and_rechecks_tensors(
    tmp_path,
):
    from fluidmix.explanation_contract import ContractError
    from fluidmix.stage_cosmos_explanation import build_stage_attempt_plan
    from fluidmix.stage_cosmos_inputs import (
        assert_stage_attempt_matches_audit,
        audit_stage_model_inputs,
        build_stage_model_inputs,
    )

    measured = _measured()
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    plan = build_stage_attempt_plan(measured, prompt)
    processor = _FakeProcessor()
    config = _input_config(processor)
    locators = _locators(tmp_path, plan)

    audit = audit_stage_model_inputs(
        plan,
        processor,
        image_locators=locators,
        config=config,
        observed_transformers_version="5.17.0",
    )

    assert audit["model_attempt_count"] == 36
    assert len(audit["attempts"]) == 36
    assert audit["condition_counts"] == {
        "numbers_only": 12,
        "ordered_stage_visual_pairs": 12,
        "irrelevant_shuffled_images": 12,
    }
    spec = next(
        row for row in plan if row["condition_id"] == "ordered_stage_visual_pairs"
    )
    built = build_stage_model_inputs(
        processor, spec, image_locators=locators, config=config
    )
    assert built["content_order"] == ["image"] * 6 + ["text"]
    assert built["image_ids"] == spec["image_ids"]
    assert_stage_attempt_matches_audit(spec, built, audit)
    built["tensors"]["input_ids"][0, 0] = 99
    with pytest.raises(ContractError, match="differs"):
        assert_stage_attempt_matches_audit(spec, built, audit)


def test_stage_generation_rechecks_audit_and_uses_frozen_native_decoding(tmp_path):
    from fluidmix.stage_cosmos_explanation import build_stage_attempt_plan
    from fluidmix.stage_cosmos_inputs import (
        audit_stage_model_inputs,
        build_stage_model_inputs,
    )
    from fluidmix.stage_cosmos_runner import generate_from_audited_stage_inputs

    measured = _measured()
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    plan = build_stage_attempt_plan(measured, prompt)
    processor = _FakeProcessor()
    config = _input_config(processor)
    locators = _locators(tmp_path, plan)
    audit = audit_stage_model_inputs(
        plan,
        processor,
        image_locators=locators,
        config=config,
        observed_transformers_version="5.17.0",
    )
    spec = plan[0]
    built = build_stage_model_inputs(
        processor, spec, image_locators=locators, config=config
    )

    class _Model:
        device = "cpu"

        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[11, 12, 13, 151643]])

    model = _Model()
    result = generate_from_audited_stage_inputs(
        model,
        processor,
        spec,
        built,
        audit,
        config,
        inference_context=nullcontext,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 384
    assert result["response_raw"] == "raw stage completion"
    assert result["terminal_token_is_eos"] is True


def test_executor_and_scorer_preserve_fixed_denominators_and_separate_metrics():
    from fluidmix.stage_cosmos_explanation import (
        build_stage_attempt_plan,
        stage_reference_response,
    )
    from fluidmix.stage_cosmos_runner import execute_stage_attempt_plan
    from fluidmix.stage_cosmos_scoring import score_stage_attempts

    measured = _measured()
    decisions = {row["decision_id"]: row for row in measured["decision_records"]}
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    plan = build_stage_attempt_plan(measured, prompt)

    def infer(spec):
        raw = stage_reference_response(decisions[spec["decision_id"]])
        if spec["attempt_index"] == 0:
            raw = "Okay.\n" + raw.replace("Evidence:", "> **evidence** —")
        if spec["attempt_index"] == 4:
            raw = raw.replace("Evidence: NO_DIFFERENCE", "Evidence: DIFFERENCE")
        return _completion(raw)

    attempts = execute_stage_attempt_plan(plan, infer=infer)
    score = score_stage_attempts(attempts, measured, prompt)

    assert len(attempts) == 48
    assert score["matrix"] == {
        "planned": 48,
        "completed": 48,
        "errors": 0,
        "model": {"planned": 36, "completed": 36, "errors": 0},
        "control": {"planned": 12, "completed": 12, "errors": 0},
    }
    metrics = score["metrics"]
    assert metrics["legacy_raw_strict_schema"] == {"passed": 0, "total": 48}
    assert metrics["legacy_raw_strict_schema_model"] == {"passed": 0, "total": 36}
    assert metrics["exact_three_line_format"] == {"passed": 47, "total": 48}
    assert metrics["lenient_parse"] == {"passed": 48, "total": 48}
    assert metrics["strict_content_policy"] == {"passed": 47, "total": 48}
    assert metrics["required_abstention"] == {"passed": 16, "total": 16}
    assert metrics["required_abstention_model"] == {"passed": 12, "total": 12}
    assert metrics["numeric_fidelity"] == {"passed": 48, "total": 48}
    assert metrics["evidence_id_faithfulness"] == {"passed": 48, "total": 48}
    assert metrics["unit_fidelity"] == {"passed": 48, "total": 48}
    assert metrics["required_limits_present"] == {"passed": 48, "total": 48}
    assert metrics["unsupported_claims_absent"] == {"passed": 48, "total": 48}
    assert metrics["operator_control_advice_absent"] == {"passed": 48, "total": 48}
    assert metrics["relevant_irrelevant_exact_invariance"] == {
        "passed": 12,
        "total": 12,
    }
    assert metrics["template_control_content_policy"] == {"passed": 12, "total": 12}
    assert score["failures"] == [
        {
            "attempt_id": "numbers_only-DEC-01",
            "categories": ["evidence_meaning", "strict_content_policy"],
        }
    ]
    assert len(score["structured_records"]) == 47
    assert all(
        row["record"]["disposition"]
        == decisions[row["decision_id"]]["disposition"]
        for row in score["structured_records"]
    )
    assert all(
        row["record"]["stage_evidence"][0]["target_delta_e76_raw"]
        == decisions[row["decision_id"]]["stages"][0]["target_delta_e76_raw"]
        for row in score["structured_records"]
    )


def test_stage_executor_retains_36_load_errors_and_aborts_integrity_errors():
    from fluidmix.cosmos_nano_runner import FatalRunError
    from fluidmix.explanation_contract import ContractError
    from fluidmix.stage_cosmos_explanation import build_stage_attempt_plan
    from fluidmix.stage_cosmos_runner import execute_stage_attempt_plan

    measured = _measured()
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    plan = build_stage_attempt_plan(measured, prompt)

    load_failed = execute_stage_attempt_plan(
        plan,
        infer=None,
        model_load_error={"type": "OSError", "message": "load failed"},
    )

    assert sum(row["attempt_status"] == "error" for row in load_failed) == 36
    assert sum(row["attempt_status"] == "completed" for row in load_failed) == 12

    def integrity_failure(spec):
        raise ContractError("live stage tensors differ from audit")

    with pytest.raises(FatalRunError, match="integrity"):
        execute_stage_attempt_plan(plan, infer=integrity_failure)


def test_stage_scorer_rejects_raw_response_hash_tampering():
    from fluidmix.explanation_contract import ContractError
    from fluidmix.stage_cosmos_explanation import (
        build_stage_attempt_plan,
        stage_reference_response,
    )
    from fluidmix.stage_cosmos_runner import execute_stage_attempt_plan
    from fluidmix.stage_cosmos_scoring import score_stage_attempts

    measured = _measured()
    decisions = {row["decision_id"]: row for row in measured["decision_records"]}
    prompt = (ROOT / "data/cosmos_stage_prompt_v1.txt").read_text(encoding="utf-8")
    plan = build_stage_attempt_plan(measured, prompt)
    attempts = execute_stage_attempt_plan(
        plan,
        infer=lambda spec: _completion(
            stage_reference_response(decisions[spec["decision_id"]])
        ),
    )
    attempts[0]["response_raw"] += " tampered"

    with pytest.raises(ContractError, match="hash mismatch"):
        score_stage_attempts(attempts, measured, prompt)


def test_stage_config_freezes_batch_matrix_and_all_source_hashes():
    from fluidmix.explanation_contract import ContractError
    from fluidmix import stage_cosmos_contract as contract

    config = copy.deepcopy(contract._FIXED)
    config.update(
        expected_tokenizer_class="Qwen2Tokenizer",
        measured_evidence_sha256="1" * 64,
        image_locator_sha256="2" * 64,
        image_preprocessor_sha256="3" * 64,
        input_audit_sha256="4" * 64,
        source_sha256={path: "5" * 64 for path in contract.SOURCE_PATHS},
    )

    assert contract.validate_stage_cosmos_config(config) is config
    changed = copy.deepcopy(config)
    changed["batch_size"] = 2
    with pytest.raises(ContractError, match="batch_size"):
        contract.validate_stage_cosmos_config(changed)
    changed = copy.deepcopy(config)
    changed["source_sha256"].pop(contract.SOURCE_PATHS[-1])
    with pytest.raises(ContractError, match="source_sha256"):
        contract.validate_stage_cosmos_config(changed)
    candidate = copy.deepcopy(config)
    candidate["config_status"] = "audit_candidate"
    candidate["input_audit_sha256"] = "0" * 64
    assert contract.validate_stage_cosmos_audit_candidate_config(candidate) is candidate
    with pytest.raises(ContractError):
        contract.validate_stage_cosmos_config(candidate)

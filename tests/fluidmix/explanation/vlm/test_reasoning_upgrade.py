"""Iteration-1 contracts for grounded reasoning and constrained VLM decoding."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from fluidmix.explanation.vlm.constrained import (
    ConstrainedDecodingError,
    build_decoder_spec,
    build_logits_processor,
    resolve_token_budget,
    runtime_vocab_size,
)
from fluidmix.explanation.vlm.decision import (
    GENERATION_CONFIG,
    MODEL_REASONING_MODES,
    RESPONSE_FIELD_ORDER,
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    parse_model_response,
)
from fluidmix.explanation.vlm.development_materializer import crop_vial_roi

QWEN = "Qwen/Qwen3-VL-8B-Instruct"
COSMOS = "nvidia/Cosmos-Reason2-8B"
ANOMALY = "aacudad/AnomalyThink-Qwen3-VL-8B-KCR"


def test_v2_response_contract_is_reasoning_led_and_bounded() -> None:
    assert RESPONSE_FIELD_ORDER == (
        "stages", "disposition", "confidence",
        "agrees_with_deterministic", "why_agree_or_disagree",
        "physics_used", "explanation", "evidence_citations",
        "image_citations",
    )
    assert RESPONSE_SCHEMA["required"] == list(RESPONSE_FIELD_ORDER)
    stage = RESPONSE_SCHEMA["properties"]["stages"]["items"]
    assert stage["required"] == [
        "stage_id", "what_i_see", "physical_reasoning",
        "vial1_vs_references", "image_quality", "cited_image_ids",
    ]
    assert stage["properties"]["what_i_see"]["maxLength"] == 600
    assert stage["properties"]["physical_reasoning"]["maxLength"] == 600
    assert RESPONSE_SCHEMA["properties"]["why_agree_or_disagree"]["maxLength"] == 600
    assert RESPONSE_SCHEMA["properties"]["explanation"]["maxLength"] == 400
    normalized = " ".join(USER_PROMPT_TEMPLATE.lower().split())
    assert "must copy" not in normalized
    assert "authoritative" not in normalized
    assert "confidence must" not in normalized


def _valid_response() -> str:
    return (
        '{"stages":[{"stage_id":"stage-1","what_i_see":"The images are limited",'
        '"physical_reasoning":"Glare could obscure the colour",'
        '"vial1_vs_references":"cannot_tell","image_quality":"limited",'
        '"cited_image_ids":[]}],'
        '"disposition":"insufficient_evidence","confidence":0.5,'
        '"agrees_with_deterministic":"partly",'
        '"why_agree_or_disagree":"The images do not resolve the comparison",'
        '"physics_used":["refraction"],'
        '"explanation":"The images do not provide enough evidence.",'
        '"evidence_citations":["EV-1"],'
        '"image_citations":["IMG-1"]}'
    )


def test_prompt_is_domain_grounded_without_forbidden_oracle_language() -> None:
    combined = f"{SYSTEM_PROMPT}\n{USER_PROMPT_TEMPLATE}".lower()
    for required in (
        "t100", "t200", "t400", "t300", "leading slug", "vials 2 and 3",
        "meniscus", "bubbles", "foam", "settling", "specular", "refraction",
        "curved-glass", "occlusion by the clamp", "motion blur",
        "lighting drift", "cielab", "deltae76", "mad", "iqr", "fill level",
        "stratification", "reynolds number", "path length",
    ):
        assert required in combined
    for forbidden in ("beer-lambert", "simulator truth", "oracle", "test label", "file path"):
        assert forbidden not in combined


def test_response_schema_freezes_reasoning_fields_before_disposition() -> None:
    assert RESPONSE_FIELD_ORDER[:2] == ("stages", "disposition")
    assert RESPONSE_SCHEMA["required"] == list(RESPONSE_FIELD_ORDER)
    assert list(RESPONSE_SCHEMA["properties"]) == list(RESPONSE_FIELD_ORDER)
    assert MODEL_REASONING_MODES == {
        QWEN: "schema_json", COSMOS: "native_think_answer",
        ANOMALY: "native_think_answer",
    }


def test_response_schema_requires_reasoning_and_stage_quality() -> None:
    stage = RESPONSE_SCHEMA["properties"]["stages"]["items"]
    assert stage["properties"]["vial1_vs_references"] == {
        "enum": [
            "same", "slightly_different", "clearly_different", "cannot_tell",
        ]
    }
    assert stage["properties"]["image_quality"] == {
        "enum": ["good", "limited", "poor"]
    }
    assert stage["properties"]["cited_image_ids"] == {
        "type": "array", "items": {"type": "string"},
    }


def test_disposition_confidence_value_and_agreement_are_closed_fields() -> None:
    assert RESPONSE_SCHEMA["properties"]["disposition"] == {
        "enum": ["discard", "insufficient_evidence", "keep"]
    }
    assert RESPONSE_SCHEMA["properties"]["confidence"] == {
        "type": "number", "minimum": 0.0, "maximum": 1.0,
    }
    assert RESPONSE_SCHEMA["properties"]["agrees_with_deterministic"] == {
        "enum": ["yes", "no", "partly"]
    }
    assert RESPONSE_SCHEMA["properties"]["explanation"] == {
        "type": "string", "minLength": 1, "maxLength": 400,
    }


def test_prompt_requests_independent_stage_reasoning_without_copy_instructions() -> None:
    normalized = " ".join(USER_PROMPT_TEMPLATE.lower().split())
    assert "early, mid and end images" in normalized
    assert "reason about what physically explains it" in normalized
    assert "whether you agree with the deterministic rule" in normalized
    assert "must copy" not in normalized
    assert "authoritative" not in normalized


def test_model_specific_parser_requires_exactly_one_native_answer() -> None:
    expected = parse_model_response(_valid_response(), repository=QWEN)
    native = f"<think>compare stages</think><answer>{_valid_response()}</answer>"
    assert parse_model_response(native, repository=COSMOS) == expected
    fenced = f"<think>compare</think><answer>```json\n{_valid_response()}\n```</answer>"
    assert parse_model_response(fenced, repository=ANOMALY) == expected
    with pytest.raises(Exception, match="exactly one.*answer"):
        parse_model_response(native + f"<answer>{_valid_response()}</answer>", repository=COSMOS)
    with pytest.raises(Exception, match="think.*answer|native reasoning"):
        parse_model_response(_valid_response(), repository=COSMOS)
    with pytest.raises(Exception, match="JSON"):
        parse_model_response(native, repository=QWEN)


def test_decoder_specs_constrain_qwen_json_and_native_tagged_answer() -> None:
    qwen = build_decoder_spec(QWEN, RESPONSE_SCHEMA)
    assert qwen == {"kind": "json_schema", "schema": RESPONSE_SCHEMA, "any_order": False}
    native = build_decoder_spec(COSMOS, RESPONSE_SCHEMA)
    assert native["kind"] == "structural_tag"
    think, answer = native["structural_tag"]["format"]["elements"]
    assert (think["begin"], think["end"]) == ("<think>", "</think>")
    assert think["content"]["type"] == "any_text"
    assert (answer["begin"], answer["end"]) == ("<answer>", "</answer>")
    assert answer["content"] == {
        "type": "json_schema", "json_schema": RESPONSE_SCHEMA, "any_order": False,
    }


class _FakeBackend:
    version = "0.2.8"

    def __init__(self) -> None:
        self.seen = None

    def compile(self, runtime, spec):
        self.seen = (runtime, spec)
        return "grammar"

    def logits_processor(self, grammar):
        assert grammar == "grammar"
        return "processor"


def test_constrained_adapter_is_pinned_and_fail_closed() -> None:
    backend = _FakeBackend()
    runtime = object()
    processor, receipt = build_logits_processor(
        runtime, repository=COSMOS, response_schema=RESPONSE_SCHEMA, backend=backend,
    )
    assert processor == "processor"
    assert backend.seen[0] is runtime
    assert backend.seen[1]["kind"] == "structural_tag"
    assert receipt["backend"] == "xgrammar"
    assert receipt["version"] == "0.2.8"
    assert receipt["constrained_region"] == "answer_json_only"
    backend.version = "0.2.7"
    with pytest.raises(ConstrainedDecodingError, match="0.2.8"):
        build_logits_processor(
            runtime, repository=QWEN, response_schema=RESPONSE_SCHEMA, backend=backend,
        )


def test_qwen3_vl_vocab_size_uses_nested_text_config() -> None:
    runtime = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
        vocab_size=None, text_config=SimpleNamespace(vocab_size=151936),
    )))
    assert runtime_vocab_size(runtime) == 151936


def test_generation_and_processor_settings_freeze_exact_10000() -> None:
    assert GENERATION_CONFIG["max_new_tokens"] == 10000
    assert GENERATION_CONFIG["do_sample"] is False
    assert GENERATION_CONFIG["seed"] == 17
    assert GENERATION_CONFIG["processor_pixels"] == {"min_pixels": 1, "max_pixels": 16777216}
    assert GENERATION_CONFIG["torch_deterministic_algorithms"] is True
    assert GENERATION_CONFIG["cudnn_benchmark"] is False
    assert GENERATION_CONFIG["per_model_fallbacks"] == {
        QWEN: None, COSMOS: None, ANOMALY: None,
    }


def test_token_budget_incompatibility_and_proven_fallback_are_explicit() -> None:
    with pytest.raises(ConstrainedDecodingError, match="10000.*context"):
        resolve_token_budget(
            repository=COSMOS, requested=10000, prompt_tokens=5000,
            context_limit=12000, fallback=None,
        )
    budget, provenance = resolve_token_budget(
        repository=COSMOS, requested=10000, prompt_tokens=5000,
        context_limit=12000,
        fallback={"max_new_tokens": 7000, "proof_receipt_sha256": "a" * 64},
    )
    assert budget == 7000
    assert provenance == {
        "requested_max_new_tokens": 10000, "effective_max_new_tokens": 7000,
        "fallback_used": True, "fallback_proof_receipt_sha256": "a" * 64,
        "reason": "model_context_limit",
    }


def test_native_roi_crop_preserves_pixels_and_binds_full_frame() -> None:
    frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    cropped, binding = crop_vial_roi(frame, [3, 2, 11, 10])
    assert np.array_equal(cropped, frame[2:10, 3:11])
    assert binding["crop_box_xyxy"] == [3, 2, 11, 10]
    assert binding["source_native_dimensions"] == [16, 12]
    assert binding["crop_native_dimensions"] == [8, 8]
    assert len(binding["source_full_frame_sha256"]) == 64
    with pytest.raises(Exception, match="ROI"):
        crop_vial_roi(frame, [3, 2, 17, 10])

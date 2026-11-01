"""Fail-closed XGrammar adapter for WP7 stock Transformers runtimes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from collections.abc import Mapping
from typing import Any


XGRAMMAR_VERSION = "0.2.8"
XGRAMMAR_LICENSE = "Apache-2.0"
DECODER_ID = "xgrammar-0.2.8-wp7-structural-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_QWEN = "Qwen/Qwen3-VL-8B-Instruct"
_NATIVE = {
    "nvidia/Cosmos-Reason2-8B",
    "aacudad/AnomalyThink-Qwen3-VL-8B-KCR",
}


class ConstrainedDecodingError(RuntimeError):
    """The required token-level decoding constraint cannot be established."""


def runtime_vocab_size(runtime: Any) -> int:
    """Resolve Qwen3-VL's vocabulary from its nested text configuration."""
    config = getattr(getattr(runtime, "model", None), "config", None)
    value = getattr(config, "vocab_size", None)
    if value is None:
        value = getattr(getattr(config, "text_config", None), "vocab_size", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConstrainedDecodingError("runtime tokenizer/vocabulary is unavailable")
    return value


def build_decoder_spec(repository: str, response_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact one-pass grammar request for one pinned model family."""
    schema = dict(response_schema)
    if repository == _QWEN:
        return {"kind": "json_schema", "schema": schema, "any_order": False}
    if repository not in _NATIVE:
        raise ConstrainedDecodingError("model has no frozen constrained-decoding mode")
    structural_tag = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [
                {
                    "type": "tag", "begin": "<think>",
                    "content": {
                        "type": "any_text", "excludes": ["<answer>"],
                        "max_tokens": 7000, "max_chars": 65536,
                    },
                    "end": "</think>",
                },
                {
                    "type": "tag", "begin": "<answer>",
                    "content": {
                        "type": "json_schema", "json_schema": schema,
                        "any_order": False,
                    },
                    "end": "</answer>",
                },
            ],
        },
    }
    return {"kind": "structural_tag", "structural_tag": structural_tag}


class XGrammarBackend:
    """Thin import-late wrapper around the pinned optional XGrammar API."""

    def __init__(self) -> None:
        try:
            import xgrammar as xgr
        except ImportError as exc:
            raise ConstrainedDecodingError(
                "xgrammar==0.2.8 is required for constrained decoding"
            ) from exc
        try:
            self.version = importlib.metadata.version("xgrammar")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ConstrainedDecodingError("xgrammar distribution metadata is absent") from exc
        self._xgr = xgr

    def compile(self, runtime: Any, spec: Mapping[str, Any]) -> Any:
        processor = getattr(runtime, "processor", None)
        tokenizer = getattr(processor, "tokenizer", processor)
        if tokenizer is None:
            raise ConstrainedDecodingError("runtime tokenizer/vocabulary is unavailable")
        vocab_size = runtime_vocab_size(runtime)
        try:
            tokenizer_info = self._xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=vocab_size
            )
            compiler = self._xgr.GrammarCompiler(tokenizer_info)
            if spec["kind"] == "json_schema":
                return compiler.compile_json_schema(
                    json.dumps(spec["schema"], separators=(",", ":")),
                    any_order=False,
                )
            if spec["kind"] == "structural_tag":
                return compiler.compile_structural_tag(spec["structural_tag"])
        except Exception as exc:
            raise ConstrainedDecodingError(
                "XGrammar could not compile the frozen decoder specification"
            ) from exc
        raise ConstrainedDecodingError("unknown constrained-decoding specification")

    def logits_processor(self, grammar: Any) -> Any:
        try:
            return self._xgr.contrib.hf.LogitsProcessor(grammar)
        except Exception as exc:
            raise ConstrainedDecodingError(
                "XGrammar could not create a fresh Transformers logits processor"
            ) from exc


def build_logits_processor(
    runtime: Any, *, repository: str, response_schema: Mapping[str, Any],
    backend: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compile and instantiate a fresh logits processor for one generation call."""
    backend = backend or XGrammarBackend()
    if getattr(backend, "version", None) != XGRAMMAR_VERSION:
        raise ConstrainedDecodingError(
            f"xgrammar=={XGRAMMAR_VERSION} is required; version differs"
        )
    spec = build_decoder_spec(repository, response_schema)
    try:
        grammar = backend.compile(runtime, spec)
        processor = backend.logits_processor(grammar)
    except ConstrainedDecodingError:
        raise
    except Exception as exc:
        raise ConstrainedDecodingError("constrained-decoding backend failed") from exc
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    return processor, {
        "backend": "xgrammar", "version": XGRAMMAR_VERSION,
        "license": XGRAMMAR_LICENSE, "decoder_id": DECODER_ID,
        "decoder_spec_sha256": hashlib.sha256(encoded).hexdigest(),
        "reasoning_mode": (
            "schema_json" if repository == _QWEN else "native_think_answer"
        ),
        "constrained_region": (
            "whole_response_json" if repository == _QWEN else "answer_json_only"
        ),
    }


def resolve_token_budget(
    *, repository: str, requested: int, prompt_tokens: int,
    context_limit: int | None, fallback: Mapping[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    """Use 10k unless context forces one explicitly proven per-model fallback."""
    if requested != 10000:
        raise ConstrainedDecodingError("the frozen requested max_new_tokens must be 10000")
    if prompt_tokens < 0:
        raise ConstrainedDecodingError("prompt token count is invalid")
    if context_limit is None or prompt_tokens + requested <= context_limit:
        return requested, {
            "requested_max_new_tokens": requested,
            "effective_max_new_tokens": requested,
            "fallback_used": False,
            "fallback_proof_receipt_sha256": None,
            "reason": None,
        }
    if fallback is None:
        raise ConstrainedDecodingError(
            f"requested 10000 tokens exceed model context for {repository}; no proven fallback"
        )
    value = fallback.get("max_new_tokens")
    proof = fallback.get("proof_receipt_sha256")
    if (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        or value >= requested or prompt_tokens + value > context_limit
        or not isinstance(proof, str) or _SHA.fullmatch(proof) is None
    ):
        raise ConstrainedDecodingError("per-model fallback proof or token budget is invalid")
    return value, {
        "requested_max_new_tokens": requested,
        "effective_max_new_tokens": value,
        "fallback_used": True,
        "fallback_proof_receipt_sha256": proof,
        "reason": "model_context_limit",
    }

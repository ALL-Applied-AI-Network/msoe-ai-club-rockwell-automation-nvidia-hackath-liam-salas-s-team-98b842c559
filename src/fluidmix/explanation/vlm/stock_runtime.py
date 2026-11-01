"""Production-owned local-only stock runtime for the three frozen Qwen3-VL models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Any, Sequence


class ContractError(ValueError):
    pass


PINNED_MODELS = {
    "Qwen/Qwen3-VL-8B-Instruct": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "nvidia/Cosmos-Reason2-8B": "a9fae2cf89dc64db96b12860417f0eb403013bb9",
    "aacudad/AnomalyThink-Qwen3-VL-8B-KCR": "8331bc79ecf01e3466f0e6a27cec8c77f182f430",
}


@dataclass(frozen=True)
class StockRuntime:
    repository: str
    revision: str
    snapshot: Path
    processor: Any
    model: Any
    torch_module: Any
    processor_class: str
    model_class: str
    inventory: dict[str, Any]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def snapshot_inventory(snapshot: str | Path) -> dict[str, Any]:
    supplied = Path(snapshot)
    if supplied.is_symlink():
        raise ContractError("model snapshot path must not be a symlink")
    root = supplied.resolve()
    if not root.is_dir():
        raise ContractError("model snapshot must be a real directory")
    files: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = path.relative_to(root)
            if relative.parts == (".cache",):
                if path.is_symlink() or not path.is_dir():
                    raise ContractError("model snapshot .cache control path is unsafe")
                continue
            if path.is_symlink():
                raise ContractError(f"model snapshot contains a symlink: {path}")
            if path.is_dir():
                visit(path)
            elif path.is_file():
                payload = path.read_bytes()
                files.append({
                    "path": relative.as_posix(), "bytes": len(payload),
                    "sha256": _sha(payload),
                })
            else:
                raise ContractError(f"model snapshot contains a non-regular file: {path}")

    visit(root)
    files.sort(key=lambda row: row["path"])
    if not files:
        raise ContractError("model snapshot inventory is empty")
    encoded = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return {"files": files, "sha256": _sha(encoded)}


def _validate_snapshot(repository: str, revision: str, snapshot: Path) -> dict[str, Any]:
    if PINNED_MODELS.get(repository) != revision:
        raise ContractError("model must use an exact pinned repository and revision")
    inventory = snapshot_inventory(snapshot)
    config_path = snapshot / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ContractError("stock model snapshot lacks a regular config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("stock model config is not valid JSON") from error
    if not isinstance(config, dict):
        raise ContractError("stock model config must be an object")
    if config.get("model_type") != "qwen3_vl":
        raise ContractError("stock model config type differs from the exact pinned policy")
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise ContractError("stock model architecture is not Qwen3VLForConditionalGeneration")
    if config.get("auto_map") not in (None, {}):
        raise ContractError("stock model config declares forbidden remote-code auto_map")
    return inventory


def load_stock_runtime(
    repository: str, revision: str, snapshot_path: str | Path, *,
    processor_class: Any | None = None, model_class: Any | None = None,
    torch_module: Any | None = None,
) -> StockRuntime:
    supplied = Path(snapshot_path)
    inventory = _validate_snapshot(repository, revision, supplied)
    snapshot = supplied.resolve()
    if processor_class is None or model_class is None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor_class = processor_class or AutoProcessor
        model_class = model_class or Qwen3VLForConditionalGeneration
    if torch_module is None and processor_class.__module__.startswith("transformers"):
        import torch as torch_module
    common = {"trust_remote_code": False, "local_files_only": True}
    processor = processor_class.from_pretrained(snapshot, **common)
    model_kwargs = dict(common)
    if torch_module is not None:
        model_kwargs.update(torch_dtype=torch_module.bfloat16, device_map="cuda")
    model = model_class.from_pretrained(snapshot, **model_kwargs)
    model.eval()
    return StockRuntime(
        repository=repository, revision=revision, snapshot=snapshot,
        processor=processor, model=model, torch_module=torch_module,
        processor_class=type(processor).__name__,
        model_class=type(model).__name__, inventory=inventory,
    )


def _messages(prompt: str, images: Sequence[object]) -> list[dict[str, Any]]:
    from PIL import Image
    content = []
    for image in images:
        with Image.open(Path(str(getattr(image, "path")))) as opened:
            content.append({"type": "image", "image": opened.convert("RGB")})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _width(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return len(value[0])
    raise ContractError("processor batch has no two-dimensional input_ids")


def _suffix(generated: Any, width: int) -> Any:
    try:
        return generated[:, width:]
    except (IndexError, TypeError):
        if isinstance(generated, (list, tuple)) and len(generated) == 1:
            return [generated[0][width:]]
        raise ContractError("model output is not a batch of token IDs")


def _move(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if hasattr(batch, "items"):
        return {key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()}
    raise ContractError("processor output is not movable model inputs")


def _filter(batch: Any, model: Any) -> Any:
    if not hasattr(batch, "items"):
        return batch
    try:
        signature = inspect.signature(model.forward)
    except (AttributeError, TypeError, ValueError):
        return batch
    if any(item.kind is inspect.Parameter.VAR_KEYWORD
           for item in signature.parameters.values()):
        return batch
    accepted = set(signature.parameters)
    return {key: value for key, value in batch.items() if key in accepted}


def _memory(torch_module: Any) -> tuple[int | None, int | None]:
    if not _cuda_available(torch_module):
        return None, None
    return int(torch_module.cuda.max_memory_allocated()), int(torch_module.cuda.max_memory_reserved())


def _cuda_available(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    probe = getattr(cuda, "is_available", None)
    return bool(callable(probe) and probe())


def invoke_stock_runtime(
    runtime: StockRuntime, prompt: str, images: tuple[object, ...], *,
    max_new_tokens: int = 384, logits_processor: Any = None,
    processor_pixels: dict[str, int] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    prompt_width = context_limit = None
    try:
        torch_module = runtime.torch_module
        if _cuda_available(torch_module):
            torch_module.cuda.reset_peak_memory_stats()
        image_processor = getattr(runtime.processor, "image_processor", None)
        if processor_pixels is not None:
            if image_processor is None:
                raise ContractError("processor has no image_processor pixel configuration")
            for field in ("min_pixels", "max_pixels"):
                value = processor_pixels.get(field)
                if not isinstance(value, int) or value <= 0:
                    raise ContractError("processor pixel configuration is invalid")
                setattr(image_processor, field, value)
        batch = runtime.processor.apply_chat_template(
            _messages(prompt, images), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        batch = _filter(_move(batch, getattr(runtime.model, "device", "cuda")), runtime.model)
        prompt_width = _width(batch["input_ids"])
        config = getattr(runtime.model, "config", None)
        text_config = getattr(config, "text_config", None)
        context_limit = getattr(text_config, "max_position_embeddings", None)
        if context_limit is None:
            context_limit = getattr(config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and prompt_width + max_new_tokens > context_limit:
            raise ContractError("requested generation exceeds model context limit")
        inference_mode = getattr(torch_module, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        if _cuda_available(torch_module):
            torch_module.cuda.synchronize()
        started = time.monotonic()
        kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens}
        if logits_processor is not None:
            kwargs["logits_processor"] = [logits_processor]
        with context:
            generated = runtime.model.generate(**batch, **kwargs)
        if _cuda_available(torch_module):
            torch_module.cuda.synchronize()
        finished = time.monotonic()
        suffix = _suffix(generated, prompt_width)
        decoded = runtime.processor.batch_decode(
            suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
            raise ContractError("processor batch_decode must return one string")
        allocated, reserved = _memory(torch_module)
        return {
            "case_id": "stock-runtime-attempt", "attempt_status": "completed",
            "status": "completed", "response_raw": decoded[0],
            "response_utf8_sha256": _sha(decoded[0].encode()),
            "generated_token_count": _width(suffix), "prompt_token_count": prompt_width,
            "model_context_limit": context_limit,
            "elapsed_seconds": round(finished - started, 6),
            "peak_cuda_memory_allocated_bytes": allocated,
            "peak_cuda_memory_reserved_bytes": reserved,
            "peak_memory_bytes": 0 if allocated is None else allocated,
            "processor_telemetry": None,
            "telemetry_unavailable_reason": None if allocated is not None else "cuda_unavailable",
            "error_type": None, "error": None,
        }
    except Exception as error:
        allocated, reserved = _memory(runtime.torch_module)
        return {
            "case_id": "stock-runtime-attempt", "attempt_status": "error",
            "status": "error", "response_raw": None, "response_utf8_sha256": None,
            "generated_token_count": None, "prompt_token_count": prompt_width,
            "model_context_limit": context_limit,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "peak_cuda_memory_allocated_bytes": allocated,
            "peak_cuda_memory_reserved_bytes": reserved,
            "peak_memory_bytes": 0 if allocated is None else allocated,
            "processor_telemetry": None, "error_type": type(error).__name__,
            # Exception messages from compiler/model stacks can contain temporary
            # paths or credentials. Retain a bounded actionable category only.
            "error": f"{type(error).__name__}: model invocation failed"[:160],
        }


__all__ = ["ContractError", "StockRuntime", "snapshot_inventory", "load_stock_runtime", "invoke_stock_runtime"]

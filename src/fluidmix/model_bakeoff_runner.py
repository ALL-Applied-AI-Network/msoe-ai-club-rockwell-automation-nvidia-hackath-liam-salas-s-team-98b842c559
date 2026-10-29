"""Pinned model adapters for the frozen real-validation explanation bake-off.

This module deliberately stops at model invocation.  It validates the frozen
case/image contract, runs each model with the same messages and greedy decoding,
and retains raw completions, errors, latency, and CUDA memory telemetry.  It has
no scorer, threshold, or disposition logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import time
from contextlib import nullcontext
from typing import Any, Callable, Iterable


class ContractError(ValueError):
    """The frozen input or model-selection contract is invalid."""


NEMOTRON_3_REVISION = "e5e9932441de940c9a62185c870ea5bcd4cd24e2"
NEMOTRON_VL_REPOSITORY = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
NEMOTRON_VL_ADAPTER_RECEIPT = "inn51_lora_training_receipt.v1"


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    revision: str
    expected_model_types: tuple[str, ...] = ()
    stock_runtime: bool = False
    trust_remote_code: bool = True


MODEL_SPECS = {
    "nvidia/Cosmos3-Nano": ModelSpec(
        "nvidia/Cosmos3-Nano",
        "e59a53c25979a090fa8706c9acc0c254a6e89b92",
        ("cosmos3_omni",),
    ),
    "nvidia/Cosmos-Reason1-7B": ModelSpec(
        "nvidia/Cosmos-Reason1-7B",
        "375e24000b24baed78f4618d3dd779e47cd96323",
        ("qwen2_5_vl",),
    ),
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16": ModelSpec(
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
        "ca9543b126e8bf3176916d3d305ccc415f89fd4d",
        ("NemotronH_Nano_VL_V2",),
    ),
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": ModelSpec(
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        NEMOTRON_3_REVISION,
        ("NemotronH_Nano_Omni_Reasoning_V3",),
    ),
    "Qwen/Qwen3-VL-8B-Instruct": ModelSpec(
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        ("qwen3_vl",),
        stock_runtime=True,
        trust_remote_code=False,
    ),
    "aacudad/AnomalyThink-Qwen3-VL-8B-KCR": ModelSpec(
        "aacudad/AnomalyThink-Qwen3-VL-8B-KCR",
        "8331bc79ecf01e3466f0e6a27cec8c77f182f430",
        ("qwen3_vl",),
        stock_runtime=True,
        trust_remote_code=False,
    ),
    "nvidia/Cosmos-Reason2-8B": ModelSpec(
        "nvidia/Cosmos-Reason2-8B",
        "a9fae2cf89dc64db96b12860417f0eb403013bb9",
        ("qwen3_vl",),
        stock_runtime=True,
        trust_remote_code=False,
    ),
}


@dataclass(frozen=True)
class ImageInput:
    path: Path
    sha256: str
    image_id: str | None = None


@dataclass(frozen=True)
class CaseInput:
    case_id: str
    prompt: str
    images: tuple[ImageInput, ...]


@dataclass(frozen=True)
class BakeoffPlan:
    path: Path
    artifact_sha256: str
    schema_version: str
    cases: tuple[CaseInput, ...]
    prompt_sha256: str | None = None


@dataclass(frozen=True)
class StockRuntime:
    """Loaded stock Qwen runtime plus exact local snapshot provenance."""

    processor: Any
    model: Any
    torch_module: Any
    processor_class: str
    model_class: str
    inventory: dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a regular file: {path}")
    return path


def _assert_output_absent(path: str | Path) -> Path:
    """Reject occupied/symlink outputs before model loading or publication."""
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise ContractError(f"refusing to overwrite output: {output}")
    if output.parent.exists() and (output.parent.is_symlink() or not output.parent.is_dir()):
        raise ContractError(f"output parent is not a real directory: {output.parent}")
    return output


def write_json_exclusive(path: str | Path, value: Any) -> Path:
    """Atomically publish one JSON result only when its path is unoccupied."""
    output = _assert_output_absent(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.incomplete"
    if temporary.exists() or temporary.is_symlink():
        raise ContractError(f"temporary output is already occupied: {temporary}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        temporary.unlink()
        output.chmod(0o444)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    return output


def validate_model_selection(repository: str, revision: str | None) -> ModelSpec:
    """Require one exact registered repository/revision pair."""
    spec = MODEL_SPECS.get(repository)
    if spec is None or revision != spec.revision:
        raise ContractError(
            "model must use an exact pinned repository and revision; "
            f"got {repository!r} {revision!r}"
        )
    return spec


def _adapter_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a regular file: {path}")
    return path


def _adapter_hashes(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"adapter contains a symlink: {path}")
        if path.is_file():
            raw = path.read_bytes()
            files[str(path.relative_to(root))] = {
                "sha256": _sha256_bytes(raw),
                "bytes": len(raw),
            }
    return files


def validate_adapter_dir(
    adapter_dir: str | Path, repository: str, revision: str
) -> dict[str, Any]:
    """Validate one immutable LoRA adapter against the pinned 12B base."""
    if repository != NEMOTRON_VL_REPOSITORY:
        raise ContractError("adapter support is limited to the pinned 12B Nemotron VL")
    spec = validate_model_selection(repository, revision)
    root = Path(adapter_dir).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"adapter directory must be a real directory: {root}")
    config_path = _adapter_regular(root / "adapter_config.json", "adapter config")
    receipt_path = _adapter_regular(root / "training-receipt.json", "adapter training receipt")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("adapter config and training receipt must be valid JSON") from exc
    if not isinstance(config, dict) or config.get("base_model_name_or_path") != repository:
        raise ContractError("adapter base model is not the pinned 12B Nemotron VL")
    if config.get("revision") not in {None, revision}:
        raise ContractError("adapter base revision does not match the pinned model")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != NEMOTRON_VL_ADAPTER_RECEIPT:
        raise ContractError("adapter training receipt schema is unsupported")
    if receipt.get("model_id") != repository or receipt.get("model_revision") != revision:
        raise ContractError("adapter receipt base model/revision does not match the pinned model")
    if receipt.get("evaluation_dataset_used") is not False:
        raise ContractError("adapter training receipt records evaluation data")
    files = _adapter_hashes(root)
    if not any(name.startswith("adapter_model.") for name in files):
        raise ContractError("adapter directory has no adapter weights")
    return {
        "schema_version": "inn51_lora_adapter.v1",
        "adapter_dir": str(root),
        "base_model_repository": spec.repository,
        "base_model_revision": spec.revision,
        "files": files,
    }


def load_adapter_model(
    model: Any, adapter_dir: str | Path, *, peft_model_class: Any | None = None
) -> Any:
    """Attach an already-validated adapter and force inference mode."""
    if peft_model_class is None:
        from peft import PeftModel

        peft_model_class = PeftModel
    adapted = peft_model_class.from_pretrained(
        model, str(adapter_dir), is_trainable=False
    )
    adapted.eval()
    return adapted


def _image_entry(raw: Any, base: Path, index: int) -> ImageInput:
    if isinstance(raw, str):
        raw_path = raw
        expected = None
    elif isinstance(raw, dict):
        raw_path = raw.get("path") or raw.get("image_path")
        expected = raw.get("sha256") or raw.get("image_sha256")
        image_id = raw.get("image_id")
    else:
        raise ContractError(f"image {index} is not an object or path")
    if isinstance(raw, str):
        image_id = None
    if not isinstance(raw_path, str) or not raw_path:
        raise ContractError(f"image {index} has no path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    path = _regular_file(path.resolve(), f"image {index}")
    observed = _sha256_bytes(path.read_bytes())
    if not isinstance(expected, str) or expected != observed:
        raise ContractError(f"image {index} SHA-256 mismatch: {path}")
    # Open and decode now, before model loading, so malformed image evidence is
    # a preflight failure and never becomes a model error.
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise ContractError(f"image {index} is not decodable: {path}") from exc
    if image_id is not None and (not isinstance(image_id, str) or not image_id):
        raise ContractError(f"image {index} has malformed image_id")
    return ImageInput(path=path, sha256=observed, image_id=image_id)


def load_plan(path: str | Path) -> BakeoffPlan:
    """Load and validate one immutable frozen real-validation input artifact."""
    path = _regular_file(Path(path).resolve(), "bake-off plan")
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ContractError("bake-off plan is not valid JSON") from exc
    if isinstance(payload, list):
        # freeze_real_validation_bakeoff.py writes its attempt plan as a
        # deterministic top-level list, while the standalone runner also
        # accepts the wrapped form for handoff artifacts.
        payload = {
            "schema_version": "real_validation_attempt_plan.v1",
            "attempts": payload,
        }
    if not isinstance(payload, dict):
        raise ContractError("bake-off plan must be a JSON object or attempt list")
    schema = payload.get("schema_version")
    if schema not in {
        "real_validation_bakeoff.v1",
        "real_validation_attempt_plan.v1",
        "real_validation_case_manifest.v1",
        "real_validation_case_spec.v1",
        "inn51_synthetic_eval_plan.v1",
    }:
        raise ContractError("unsupported bake-off plan schema")

    raw_cases = payload.get("cases")
    if raw_cases is None:
        raw_attempts = payload.get("attempts")
        if not isinstance(raw_attempts, list):
            raw_cases = None
        else:
            # ``real_validation_bakeoff.build_attempt_plan`` emits one row per
            # case/model pair. Collapse those rows only after proving that all
            # models received byte-identical prompt and image evidence.
            grouped: dict[str, dict[str, Any]] = {}
            order: list[str] = []
            for attempt in raw_attempts:
                if not isinstance(attempt, dict):
                    raise ContractError("bake-off attempts must be objects")
                case_id = attempt.get("case_id")
                if not isinstance(case_id, str) or not case_id:
                    raise ContractError("bake-off attempt has no case ID")
                images = [{
                    "path": attempt.get("image_path"),
                    "sha256": attempt.get("image_sha256"),
                    "image_id": attempt.get("image_id"),
                }]
                candidate = {
                    "case_id": case_id,
                    "prompt": attempt.get("prompt"),
                    "prompt_sha256": attempt.get("prompt_sha256"),
                    "images": images,
                }
                if case_id not in grouped:
                    grouped[case_id] = candidate
                    order.append(case_id)
                elif grouped[case_id] != candidate:
                    raise ContractError(
                        f"case {case_id} differs across model attempts"
                    )
            raw_cases = [grouped[case_id] for case_id in order]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError("bake-off plan must contain a non-empty cases list")
    top_prompt = payload.get("prompt")
    top_prompt_template = payload.get("prompt_template")
    if top_prompt_template is not None and not isinstance(top_prompt_template, str):
        raise ContractError("prompt_template must be text")
    top_prompt_hash = payload.get("prompt_sha256")
    if top_prompt is not None and not isinstance(top_prompt, str):
        raise ContractError("top-level prompt must be text")

    cases: list[CaseInput] = []
    seen: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ContractError("each bake-off case must be an object")
        case_id = raw_case.get("case_id", raw_case.get("record_id"))
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ContractError("case IDs must be non-empty and unique")
        seen.add(case_id)
        prompt = raw_case.get("prompt", top_prompt)
        if prompt is None and isinstance(top_prompt_template, str):
            if top_prompt_template.count("{{CASE_EVIDENCE_JSON}}") != 1:
                raise ContractError("prompt_template must contain one case placeholder")
            evidence = {
                key: value
                for key, value in raw_case.items()
                if key not in {"image_path", "image_sha256"}
            }
            prompt = top_prompt_template.replace(
                "{{CASE_EVIDENCE_JSON}}",
                json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
        if not isinstance(prompt, str) or not prompt:
            raise ContractError(f"case {case_id} has no prompt")
        case_prompt_hash = raw_case.get("prompt_sha256", top_prompt_hash)
        if case_prompt_hash is not None and _sha256_text(prompt) != case_prompt_hash:
            raise ContractError(f"case {case_id} does not match prompt SHA-256")
        raw_images = raw_case.get("images", raw_case.get("image_paths"))
        if raw_images is None and "image_path" in raw_case:
            raw_images = [
                {
                    "path": raw_case.get("image_path"),
                    "sha256": raw_case.get("image_sha256"),
                    "image_id": raw_case.get("image_id"),
                }
            ]
        if not isinstance(raw_images, list) or not 1 <= len(raw_images) <= 6:
            raise ContractError(f"case {case_id} must contain one to six images")
        images = tuple(
            _image_entry(entry, path.parent, index)
            for index, entry in enumerate(raw_images)
        )
        cases.append(CaseInput(case_id=case_id, prompt=prompt, images=images))

    observed_prompt_hash = top_prompt_hash
    if observed_prompt_hash is None and top_prompt is not None:
        observed_prompt_hash = _sha256_text(top_prompt)
    return BakeoffPlan(
        path=path,
        artifact_sha256=_sha256_bytes(raw_bytes),
        schema_version=schema,
        cases=tuple(cases),
        prompt_sha256=observed_prompt_hash,
    )


def build_messages(case: CaseInput) -> list[dict[str, Any]]:
    """Build the common image-then-prompt conversation for every adapter."""
    from PIL import Image

    content: list[dict[str, Any]] = []
    for image in case.images:
        with Image.open(image.path) as opened:
            content.append({"type": "image", "image": opened.convert("RGB")})
    content.append({"type": "text", "text": case.prompt})
    return [{"role": "user", "content": content}]


def _input_ids_width(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple)):
        return len(value[0])
    raise ContractError("processor batch has no two-dimensional input_ids")


def _suffix(generated: Any, prompt_width: int) -> Any:
    try:
        return generated[:, prompt_width:]
    except (IndexError, TypeError):
        if isinstance(generated, (list, tuple)) and len(generated) == 1:
            return [generated[0][prompt_width:]]
        raise ContractError("model output is not a batch of token IDs")


def _processor_telemetry(case: CaseInput, processor: Any, batch: Any) -> dict[str, Any]:
    """Record processor/image facts without mutating model or processor state."""
    telemetry: dict[str, Any] = {
        "image_count": len(case.images),
        "original_rgb_sizes": [],
        "processor_class": type(processor).__name__,
        "processor_config_class": None,
        "processor_config_sha256": None,
        "pixel_values_shape": None,
        "pixel_values_dtype": None,
        "pixel_values_sha256": None,
    }
    from PIL import Image

    for image in case.images:
        with Image.open(image.path) as opened:
            telemetry["original_rgb_sizes"].append(list(opened.convert("RGB").size))

    config = getattr(processor, "image_processor", None)
    if config is None:
        config = getattr(processor, "processor", None)
    if config is not None:
        telemetry["processor_config_class"] = type(config).__name__
        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            try:
                encoded = json.dumps(to_dict(), sort_keys=True, default=str).encode()
                telemetry["processor_config_sha256"] = _sha256_bytes(encoded)
            except (TypeError, ValueError):
                pass

    values = batch.get("pixel_values") if hasattr(batch, "get") else None
    if values is None:
        return telemetry
    shape = getattr(values, "shape", None)
    if shape is not None:
        telemetry["pixel_values_shape"] = [int(item) for item in shape]
    dtype = getattr(values, "dtype", None)
    if dtype is not None:
        telemetry["pixel_values_dtype"] = str(dtype)
    try:
        cpu = values.detach().cpu().contiguous() if hasattr(values, "detach") else values
        if hasattr(cpu, "numpy"):
            cpu = cpu.numpy()
        try:
            import numpy as np
            cpu = np.ascontiguousarray(cpu)
            payload = cpu.tobytes()
        except ImportError:
            payload = memoryview(cpu).tobytes()
        telemetry["pixel_values_sha256"] = _sha256_bytes(payload)
    except (AttributeError, TypeError, ValueError):
        pass
    return telemetry


def _token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[-1])
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return len(value[0])
    raise ContractError("model suffix is not a one-row token batch")


def _cuda_memory(torch_module: Any) -> tuple[int | None, int | None]:
    if torch_module is None or not getattr(torch_module.cuda, "is_available", lambda: False)():
        return None, None
    return (
        int(torch_module.cuda.max_memory_allocated()),
        int(torch_module.cuda.max_memory_reserved()),
    )


def _reset_cuda_memory(torch_module: Any) -> None:
    if torch_module is not None and getattr(torch_module.cuda, "is_available", lambda: False)():
        torch_module.cuda.reset_peak_memory_stats()


def _synchronize_cuda(torch_module: Any) -> bool:
    if torch_module is None or not getattr(torch_module.cuda, "is_available", lambda: False)():
        return False
    torch_module.cuda.synchronize()
    return True


def _move_inputs(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if hasattr(batch, "items"):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    raise ContractError("processor output is not movable model inputs")


def _filter_forward_inputs(batch: Any, model: Any) -> Any:
    """Drop processor-only fields rejected by a custom model ``forward``.

    Nemotron Nano 2's processor emits ``num_patches`` for its image path, but
    its pinned custom forward signature does not consume that bookkeeping
    field.  Transformers otherwise passes it through ``generate`` and raises
    before the first token.  Models with ``**kwargs`` retain the complete
    processor batch because those adapters may intentionally consume extras.
    """
    if not hasattr(batch, "items"):
        return batch
    try:
        signature = inspect.signature(model.forward)
    except (AttributeError, TypeError, ValueError):
        return batch
    parameters = signature.parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return batch
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {key: value for key, value in batch.items() if key in accepted}


def _error_row(
    case: CaseInput,
    started: float,
    exc: BaseException,
    clock: Callable[[], float],
    torch_module: Any = None,
    attempt_index: int | None = None,
    processor_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allocated, reserved = _cuda_memory(torch_module)
    return {
        "attempt_index": attempt_index,
        "case_id": case.case_id,
        "attempt_status": "error",
        "status": "error",
        "response_utf8_sha256": None,
        "response_raw": None,
        "generated_token_count": None,
        "elapsed_seconds": round(clock() - started, 6),
        "peak_cuda_memory_allocated_bytes": allocated,
        "peak_cuda_memory_reserved_bytes": reserved,
        "peak_memory_bytes": 0 if allocated is None else allocated,
        "processor_telemetry": processor_telemetry,
        "error_type": type(exc).__name__,
        "error": str(exc)[:4000],
    }


def run_attempts(
    plan: BakeoffPlan,
    processor: Any,
    model: Any,
    *,
    max_new_tokens: int = 384,
    torch_module: Any = None,
    clock: Callable[[], float] = time.monotonic,
) -> list[dict[str, Any]]:
    """Run every frozen case once with greedy decoding and retain raw results."""
    if max_new_tokens <= 0:
        raise ContractError("max_new_tokens must be positive")
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(plan.cases):
        started = clock()
        processor_telemetry = None
        try:
            _reset_cuda_memory(torch_module)
            messages = build_messages(case)
            batch = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            processor_telemetry = _processor_telemetry(case, processor, batch)
            batch = _move_inputs(batch, getattr(model, "device", "cuda"))
            batch = _filter_forward_inputs(batch, model)
            context = (
                torch_module.inference_mode()
                if torch_module is not None and hasattr(torch_module, "inference_mode")
                else nullcontext()
            )
            synchronized = _synchronize_cuda(torch_module)
            started = clock()
            with context:
                try:
                    generated = model.generate(
                        **batch,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                    )
                finally:
                    if synchronized:
                        _synchronize_cuda(torch_module)
            generation_finished = clock()
            suffix = _suffix(generated, _input_ids_width(batch["input_ids"]))
            decoded = processor.batch_decode(
                suffix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
                raise ContractError("processor batch_decode must return one string")
            allocated, reserved = _cuda_memory(torch_module)
            rows.append(
                {
                    "case_id": case.case_id,
                    "attempt_status": "completed",
                    "status": "completed",
                    "response_raw": decoded[0],
                    "response_utf8_sha256": _sha256_text(decoded[0]),
                    "generated_token_count": _token_count(suffix),
                    "elapsed_seconds": round(generation_finished - started, 6),
                    "peak_cuda_memory_allocated_bytes": allocated,
                    "peak_cuda_memory_reserved_bytes": reserved,
                    "peak_memory_bytes": 0 if allocated is None else allocated,
                    "processor_telemetry": processor_telemetry,
                    "telemetry_unavailable_reason": (
                        None if synchronized else "cuda_unavailable"
                    ),
                    "error_type": None,
                    "error": None,
                }
            )
        except Exception as exc:
            rows.append(
                _error_row(
                    case,
                    started,
                    exc,
                    clock,
                    torch_module,
                    index,
                    processor_telemetry,
                )
            )
    return rows


def _bind_attempt_metadata(
    rows: list[dict[str, Any]], plan: BakeoffPlan, repository: str
) -> list[dict[str, Any]]:
    if len(rows) != len(plan.cases):
        raise ContractError("runner returned the wrong number of attempts")
    bound: list[dict[str, Any]] = []
    for index, (row, case) in enumerate(zip(rows, plan.cases, strict=True)):
        bound.append(
            {
                **row,
                "attempt_index": index,
                "case_id": case.case_id,
                "model_id": repository,
                "prompt_sha256": _sha256_text(case.prompt),
                "ordered_image_ids": [image.image_id for image in case.images],
                "ordered_image_sha256": [image.sha256 for image in case.images],
                # Retain the historical one-image projection for legacy plans.
                "image_id": case.images[0].image_id,
                "image_sha256": case.images[0].sha256,
            }
        )
    return bound


def snapshot_inventory(snapshot: str | Path) -> dict[str, Any]:
    """Hash immutable repository payload bytes, excluding local HF cache controls."""
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
                # huggingface_hub writes acquisition timestamps and metadata here.
                # They are local cache controls, not immutable repository payload.
                continue
            if path.is_symlink():
                raise ContractError(f"model snapshot contains a symlink: {path}")
            if path.is_dir():
                visit(path)
                continue
            if not path.is_file():
                raise ContractError(f"model snapshot contains a non-regular file: {path}")
            payload = path.read_bytes()
            files.append({
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": _sha256_bytes(payload),
            })

    visit(root)
    files.sort(key=lambda row: row["path"])
    if not files:
        raise ContractError("model snapshot inventory is empty")
    encoded = json.dumps(
        files, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {"files": files, "sha256": _sha256_bytes(encoded)}


def _validate_stock_snapshot(spec: ModelSpec, snapshot: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not spec.stock_runtime or spec.trust_remote_code:
        raise ContractError("selected model does not use the stock no-remote-code runtime")
    inventory = snapshot_inventory(snapshot)
    config_path = snapshot / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ContractError("stock model snapshot lacks a regular config.json")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("stock model config is not valid JSON") from exc
    if not isinstance(config, dict):
        raise ContractError("stock model config must be an object")
    if config.get("model_type") not in spec.expected_model_types:
        raise ContractError("stock model config type differs from the exact pinned policy")
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise ContractError("stock model architecture is not Qwen3VLForConditionalGeneration")
    if config.get("auto_map") not in (None, {}):
        raise ContractError("stock model config declares forbidden remote-code auto_map")
    return inventory, config


def load_stock_runtime(
    repository: str,
    revision: str,
    snapshot_path: str | Path,
    *,
    processor_class: Any | None = None,
    model_class: Any | None = None,
    torch_module: Any | None = None,
) -> StockRuntime:
    """Load one exact local Qwen3-VL snapshot with remote code disabled."""
    spec = validate_model_selection(repository, revision)
    supplied_snapshot = Path(snapshot_path)
    inventory, _config = _validate_stock_snapshot(spec, supplied_snapshot)
    snapshot = supplied_snapshot.resolve()
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
        processor=processor,
        model=model,
        torch_module=torch_module,
        processor_class=type(processor).__name__,
        model_class=type(model).__name__,
        inventory=inventory,
    )


def invoke_stock_runtime(
    runtime: StockRuntime,
    prompt: str,
    images: tuple[ImageInput, ...],
    *,
    max_new_tokens: int = 384,
) -> dict[str, Any]:
    """Invoke a loaded stock runtime once on all supplied ordered images."""
    case = CaseInput(case_id="stock-runtime-attempt", prompt=prompt, images=images)
    plan = BakeoffPlan(
        path=Path("<controller>"), artifact_sha256=_sha256_text(prompt),
        schema_version="controller-stock-runtime.v1", cases=(case,),
    )
    return run_attempts(
        plan, runtime.processor, runtime.model,
        max_new_tokens=max_new_tokens, torch_module=runtime.torch_module,
    )[0]


def _snapshot_metadata(snapshot: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name in ("README.md", "config.json"):
        path = snapshot / name
        if not path.is_file():
            continue
        raw = path.read_bytes()
        entry: dict[str, Any] = {"sha256": _sha256_bytes(raw), "bytes": len(raw)}
        if name == "config.json":
            try:
                config = json.loads(raw)
                entry["model_type"] = config.get("model_type")
                entry["architectures"] = config.get("architectures")
            except json.JSONDecodeError:
                entry["config_error"] = "invalid_json"
        metadata[name] = entry
    return metadata


def resolve_snapshot(spec: ModelSpec, model_dir: str | Path | None = None) -> Path:
    """Resolve an exact local snapshot without downloading anything."""
    if model_dir is not None:
        supplied = Path(model_dir)
        if supplied.is_symlink():
            raise ContractError("model path must not be a symlink")
        snapshot = _regular_file(supplied.resolve(), "model path") if supplied.is_file() else supplied.resolve()
        if not snapshot.is_dir() or snapshot.is_symlink():
            raise ContractError("model path must be a real snapshot directory")
        return snapshot
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                spec.repository,
                revision=spec.revision,
                local_files_only=True,
            )
        )
    except Exception as exc:
        raise ContractError(
            f"exact local HF snapshot is unavailable for {spec.repository}@{spec.revision}"
        ) from exc
    if not snapshot.is_dir():
        raise ContractError("HF snapshot path is not a directory")
    return snapshot


def preflight(
    repository: str,
    revision: str,
    plan: BakeoffPlan,
    model_dir: str | Path | None = None,
    adapter_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check pins, frozen inputs, and local model metadata without loading weights."""
    spec = validate_model_selection(repository, revision)
    snapshot = resolve_snapshot(spec, model_dir)
    metadata = _snapshot_metadata(snapshot)
    model_type = metadata.get("config.json", {}).get("model_type")
    if spec.expected_model_types and model_type not in spec.expected_model_types:
        raise ContractError(
            f"model config type {model_type!r} is not expected for {spec.repository}"
        )
    inventory = None
    if spec.stock_runtime:
        inventory, _config = _validate_stock_snapshot(spec, snapshot)
    result = {
        "schema_version": "model_bakeoff_preflight.v1",
        "status": "ready",
        "model_repository": spec.repository,
        "model_revision": spec.revision,
        "snapshot_path": str(snapshot),
        "model_metadata": metadata,
        "case_count": len(plan.cases),
        "artifact_sha256": plan.artifact_sha256,
        "platform_machine": platform.machine(),
    }
    if inventory is not None:
        result["snapshot_inventory"] = inventory
        result["trust_remote_code"] = False
        result["local_files_only"] = True
    if adapter_dir is not None:
        result["adapter"] = validate_adapter_dir(adapter_dir, repository, revision)
    return result


def _load_runtime(
    spec: ModelSpec,
    snapshot: Path,
    torch_module: Any,
    adapter_dir: str | Path | None = None,
) -> tuple[Any, Any, str, str]:
    if spec.stock_runtime:
        runtime = load_stock_runtime(
            spec.repository, spec.revision, snapshot, torch_module=torch_module
        )
        return (
            runtime.processor,
            runtime.model,
            runtime.processor_class,
            runtime.model_class,
        )
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        snapshot, trust_remote_code=True, local_files_only=True
    )
    classes = (
        (AutoModelForImageTextToText, AutoModelForCausalLM)
        if spec.repository == "nvidia/Cosmos3-Nano"
        else (AutoModelForCausalLM, AutoModelForImageTextToText)
    )
    model_class = classes[0]
    try:
        model = model_class.from_pretrained(
            snapshot,
            trust_remote_code=True,
            local_files_only=True,
            # ``torch_dtype`` is the Transformers loading kwarg.  The
            # Nemotron custom model constructor does not accept the newer
            # ``dtype`` alias, so using it here would fail before weights are
            # allocated even though the pinned runtime is otherwise ready.
            torch_dtype=torch_module.bfloat16,
            device_map="cuda",
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        model_class = classes[1]
        model = model_class.from_pretrained(
            snapshot,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_module.bfloat16,
            device_map="cuda",
        )
    model.eval()
    if adapter_dir is not None:
        model = load_adapter_model(model, adapter_dir)
        model.eval()
    return processor, model, type(processor).__name__, type(model).__name__


def run_model(
    repository: str,
    revision: str,
    plan: BakeoffPlan,
    *,
    max_new_tokens: int = 384,
    model_dir: str | Path | None = None,
    adapter_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load one exact model and execute the frozen plan, preserving failures."""
    spec = validate_model_selection(repository, revision)
    adapter_metadata = (
        validate_adapter_dir(adapter_dir, repository, revision)
        if adapter_dir is not None
        else None
    )
    started = time.monotonic()
    torch_module = None
    try:
        snapshot = resolve_snapshot(spec, model_dir)
        import torch as torch_module

        processor, model, processor_class, model_class = _load_runtime(
            spec, snapshot, torch_module, adapter_dir=adapter_dir
        )
        load_error = None
    except Exception as exc:
        snapshot = None
        processor = model = None
        processor_class = model_class = None
        load_error = {"type": type(exc).__name__, "message": str(exc)[:4000]}
    load_seconds = round(time.monotonic() - started, 6)
    load_allocated, load_reserved = _cuda_memory(torch_module)
    if load_error is not None:
        attempts = [
            {
                "attempt_index": index,
                "case_id": case.case_id,
                "attempt_status": "error",
                "status": "error",
                "response_utf8_sha256": None,
                "response_raw": None,
                "generated_token_count": None,
                "elapsed_seconds": 0.0,
                "peak_cuda_memory_allocated_bytes": None,
                "peak_cuda_memory_reserved_bytes": None,
                "peak_memory_bytes": 0,
                "error_type": load_error["type"],
                "error": load_error["message"],
            }
            for index, case in enumerate(plan.cases)
        ]
    else:
        attempts = run_attempts(
            plan,
            processor,
            model,
            max_new_tokens=max_new_tokens,
            torch_module=torch_module,
        )
    attempts = _bind_attempt_metadata(attempts, plan, spec.repository)
    result = {
        "schema_version": "model_bakeoff_raw.v1",
        "model_repository": spec.repository,
        "model_revision": spec.revision,
        "artifact_sha256": plan.artifact_sha256,
        "model_snapshot": str(snapshot) if snapshot is not None else None,
        "processor_class": processor_class,
        "model_class": model_class,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "batch_size": 1,
        "model_load_seconds": load_seconds,
        "model_load_peak_cuda_memory_allocated_bytes": load_allocated,
        "model_load_peak_cuda_memory_reserved_bytes": load_reserved,
        "model_load_error": load_error,
        "attempts": attempts,
    }
    if adapter_metadata is not None:
        result["adapter"] = adapter_metadata
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--revision",
        help="exact pinned commit; defaults to the registry pin for --model",
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help="optional validated INN-51 LoRA adapter; only supported for pinned 12B Nemotron VL",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        revision = args.revision or MODEL_SPECS[args.model].revision
        plan = load_plan(args.plan)
        if args.out is not None:
            _assert_output_absent(args.out)
        if args.preflight:
            value = preflight(args.model, revision, plan, args.model_dir, args.adapter_dir)
        else:
            if args.out is None:
                raise ContractError("--out is required unless --preflight is used")
            value = run_model(
                args.model,
                revision,
                plan,
                max_new_tokens=args.max_new_tokens,
                model_dir=args.model_dir,
                adapter_dir=args.adapter_dir,
            )
        encoded = json.dumps(value, indent=2, sort_keys=True)
        if args.out is None:
            print(encoded)
        else:
            write_json_exclusive(args.out, value)
            print(encoded)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

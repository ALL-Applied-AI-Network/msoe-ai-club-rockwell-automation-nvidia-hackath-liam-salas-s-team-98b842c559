#!/usr/bin/env python3
"""Run the frozen 72-row Varied V3 Cosmos3-Nano matrix locally."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _driver() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    rows = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(rows) != 1 or not rows[0]:
        raise ValueError("could not record one NVIDIA driver version")
    return rows[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--attempt-plan", type=Path, required=True)
    parser.add_argument("--image-locator", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from fluidmix.explanation.vlm.cosmos_contract_runner import (
        publish_immutable_json,
        require_output_absent,
        verify_git_freeze,
        verify_runtime_environment,
        verify_source_hashes,
    )
    from fluidmix.explanation.vlm.cosmos_nano_model_view import verify_model_view
    from fluidmix.explanation.vlm.cosmos_nano_runner import FatalRunError, load_nano_model
    from fluidmix.explanation.template.explanation_contract import ContractError
    from fluidmix.explanation.vlm.cosmos_contract import (
        SOURCE_PATHS,
        validate_v3_cosmos_config,
    )
    from fluidmix.explanation.vlm.cosmos_inputs import (
        build_v3_model_input_batch,
        generate_from_audited_v3_input_batch,
        verify_tracked_v3_input_audit,
    )
    from evaluation.scoring.cosmos_plan_artifact import load_v3_attempt_plan_artifact
    from fluidmix.explanation.vlm.cosmos_runtime import execute_v3_attempt_plan

    config = _load(args.config)
    validate_v3_cosmos_config(config)
    prompt_path = ROOT / "data/cosmos_f2_varied_prompt_v1.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    for path, expected, role in (
        (args.decisions, config["decision_artifact_sha256"], "decisions"),
        (args.attempt_plan, config["attempt_plan_sha256"], "attempt plan"),
        (args.image_locator, config["image_locator_sha256"], "image locator"),
        (prompt_path, config["prompt_template_sha256"], "prompt"),
        (args.input_audit, config["input_audit_sha256"], "input audit"),
    ):
        if _sha(path) != expected:
            raise ValueError(f"V3 {role} SHA-256 mismatch")
    decisions = _load(args.decisions)
    locator = _load(args.image_locator)
    if not isinstance(locator, dict) or not isinstance(locator.get("images"), dict):
        raise ValueError("V3 image locator artifact is malformed")
    plan_artifact, plan_sha = load_v3_attempt_plan_artifact(
        args.attempt_plan,
        decisions,
        prompt,
        decision_artifact_sha256=config["decision_artifact_sha256"],
        image_locator_sha256=config["image_locator_sha256"],
    )
    if plan_sha != config["attempt_plan_sha256"]:
        raise ValueError("V3 recomputed attempt-plan SHA-256 mismatch")
    verify_model_view(args.model_dir, args.model_manifest, config)
    runtime_versions = verify_runtime_environment(args.runtime_lock, config)
    source_hashes = verify_source_hashes(ROOT, config["source_sha256"], SOURCE_PATHS)
    head = verify_git_freeze(ROOT, config["freeze_tag"])
    require_output_absent(ROOT, config["output_path"])
    if transformers.__version__ != config["runtime_versions"]["transformers"]:
        raise ValueError("V3 imported Transformers version differs from the pin")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    audit = verify_tracked_v3_input_audit(
        args.input_audit,
        plan_artifact["attempts"],
        processor,
        image_locators=locator["images"],
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("V3 Nano run requires CUDA with bf16 support")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    started_at = _utc()
    load_started = time.monotonic()
    model = None
    model_load_error = None
    try:
        model = load_nano_model(
            AutoModelForImageTextToText, args.model_dir, torch, config
        )
        torch.cuda.synchronize()
        model_load_seconds = round(time.monotonic() - load_started, 6)
    except (ContractError, FatalRunError):
        raise
    except Exception as exc:
        if isinstance(exc, MemoryError) or "OutOfMemory" in type(exc).__name__:
            raise FatalRunError("fatal memory exhaustion while loading V3 Nano") from exc
        model_load_seconds = round(time.monotonic() - load_started, 6)
        model_load_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}

    model_class = attention = generation = None
    parameter_devices: list[str] = []
    if model is not None:
        model_class = type(model).__name__
        attention = getattr(model.config, "_attn_implementation", None)
        generation = {
            "eos_token_id": model.generation_config.eos_token_id,
            "pad_token_id": model.generation_config.pad_token_id,
            "bos_token_id": model.generation_config.bos_token_id,
            "do_sample": model.generation_config.do_sample,
        }
        if generation["eos_token_id"] != config["native_eos_token_ids"]:
            raise ValueError("V3 Nano native EOS configuration changed")
        if generation["pad_token_id"] != config["native_pad_token_id"]:
            raise ValueError("V3 Nano native pad configuration changed")
        parameter_devices = sorted({str(p.device) for p in model.parameters()})

    completed = 0

    def infer_batch(specs):
        nonlocal completed
        print(
            f"V3 model batch starting at {completed + 1}/54: "
            f"{specs[0]['condition_id']} ({len(specs)} rows)",
            flush=True,
        )
        built = build_v3_model_input_batch(
            processor,
            specs,
            image_locators=locator["images"],
            config=config,
        )
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        result = generate_from_audited_v3_input_batch(
            model,
            processor,
            specs,
            built,
            audit,
            config,
            inference_context=torch.inference_mode,
            move_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        completed += len(specs)
        return result

    attempts = execute_v3_attempt_plan(
        plan_artifact["attempts"],
        infer_batch=infer_batch if model is not None else None,
        batch_size=config["batch_size"],
        model_load_error=model_load_error,
    )
    result = {
        "schema_version": "f2_varied_cosmos3_nano_raw_results.v1",
        "experiment_id": config["experiment_id"],
        "provenance": {
            "freeze_tag": config["freeze_tag"],
            "head_commit": head,
            "git_clean": True,
            "config_sha256": _sha(args.config),
            "decision_artifact_sha256": config["decision_artifact_sha256"],
            "attempt_plan_sha256": plan_sha,
            "image_locator_sha256": config["image_locator_sha256"],
            "prompt_template_sha256": config["prompt_template_sha256"],
            "input_audit_sha256": config["input_audit_sha256"],
            "batch_size_smoke_sha256": config["batch_size_smoke_sha256"],
            "source_sha256": source_hashes,
            "model_repository": config["model_repository"],
            "model_revision": config["model_revision"],
            "seed": config["seed"],
            "do_sample": config["do_sample"],
            "max_new_tokens": config["max_new_tokens"],
            "batch_size": config["batch_size"],
            "dtype": config["dtype"],
            "python": platform.python_version(),
            "platform_machine": platform.machine(),
            "runtime_versions": runtime_versions,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__,
            "model_class": model_class,
            "attention_implementation": attention,
            "generation_config": generation,
            "parameter_devices": parameter_devices,
            "model_load_seconds": model_load_seconds,
            "model_load_error": model_load_error,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_version": torch.version.cuda,
            "driver_version": _driver(),
            "started_at_utc": started_at,
            "finished_at_utc": _utc(),
        },
        "metrics_status": "not_scored_raw_outputs_only",
        "attempts": attempts,
    }
    output = publish_immutable_json(ROOT, config["output_path"], result)
    finished = sum(row["attempt_status"] == "completed" for row in attempts)
    print(
        f"published 72 raw outcomes to {output}; completed={finished}, "
        f"errors={72 - finished}; raw responses not printed"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

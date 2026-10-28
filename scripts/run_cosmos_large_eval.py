#!/usr/bin/env python3
"""Run the frozen local Cosmos3-Nano matched matrix exactly once."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_contract_runner import (  # noqa: E402
    build_attempt_plan,
    publish_immutable_json,
    require_output_absent,
    sha256_path,
    verify_git_freeze,
    verify_runtime_environment,
    verify_source_hashes,
)
from fluidmix.cosmos_nano_contract import (  # noqa: E402
    SOURCE_PATHS,
    validate_nano_eval_config,
)
from fluidmix.cosmos_nano_input_audit import (  # noqa: E402
    build_model_inputs,
    verify_tracked_input_audit,
)
from fluidmix.cosmos_nano_model_view import verify_model_view  # noqa: E402
from fluidmix.cosmos_nano_runner import (  # noqa: E402
    FatalRunError,
    execute_nano_attempt_plan,
    generate_from_audited_inputs,
    load_nano_model,
)
from fluidmix.explanation_contract import (  # noqa: E402
    ContractError,
    validate_fixture_set,
)


FIXTURE_PATH = ROOT / "data/cosmos_contract_fixtures_v1.json"
PROMPT_PATH = ROOT / "data/cosmos_operator_prompt_v1.txt"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    return parser.parse_args()


def _driver_version():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
    )
    value = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(value) != 1 or not value[0]:
        raise ContractError("could not record one current NVIDIA driver version")
    return value[0]


def _generation_summary(model):
    generation = model.generation_config
    return {
        "eos_token_id": generation.eos_token_id,
        "pad_token_id": generation.pad_token_id,
        "bos_token_id": generation.bos_token_id,
        "do_sample": generation.do_sample,
    }


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_nano_eval_config(config)
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    validate_fixture_set(fixtures)
    if sha256_path(FIXTURE_PATH) != config["fixture_set_sha256"]:
        raise ContractError("fixture-set SHA-256 mismatch")
    if sha256_path(PROMPT_PATH) != config["prompt_sha256"]:
        raise ContractError("prompt SHA-256 mismatch")
    if sha256_path(args.image) != config["image_sha256"]:
        raise ContractError("fixed image SHA-256 mismatch")
    config_sha256 = sha256_path(args.config)
    verify_model_view(args.model_dir, args.model_manifest, config)
    runtime_versions = verify_runtime_environment(args.runtime_lock, config)
    source_hashes = verify_source_hashes(
        ROOT, config["source_sha256"], SOURCE_PATHS
    )
    head_commit = verify_git_freeze(ROOT, config["freeze_tag"])
    require_output_absent(ROOT, config["output_path"])

    from PIL import Image
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != config["runtime_versions"]["transformers"]:
        raise ContractError("imported Transformers version differs from the frozen pin")
    with Image.open(args.image) as image:
        image.load()
        if image.mode != "RGB" or image.size != (640, 480):
            raise ContractError("fixed image must be RGB 640x480")
    plan = build_attempt_plan(
        fixtures, config, PROMPT_PATH.read_text(encoding="utf-8")
    )
    processor = AutoProcessor.from_pretrained(
        args.model_dir, local_files_only=True
    )
    audit = verify_tracked_input_audit(
        args.input_audit,
        plan,
        processor,
        image_path=args.image,
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("CUDA with bf16 support is required")
    device_name = torch.cuda.get_device_name(0)
    compute_capability = list(torch.cuda.get_device_capability(0))
    driver_version = _driver_version()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    started_at = utc_now()
    model = None
    model_load_error = None
    model_load_seconds = 0.0
    load_started = time.monotonic()
    try:
        model = load_nano_model(
            AutoModelForImageTextToText,
            args.model_dir,
            torch,
            config,
        )
        torch.cuda.synchronize()
        model_load_seconds = round(time.monotonic() - load_started, 6)
    except (ContractError, FatalRunError):
        raise
    except Exception as exc:
        if isinstance(exc, MemoryError) or "OutOfMemory" in type(exc).__name__:
            raise FatalRunError("fatal memory exhaustion while loading Nano") from exc
        model_load_seconds = round(time.monotonic() - load_started, 6)
        model_load_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}

    model_class = None
    attention_implementation = None
    generation_config = None
    parameter_devices = []
    if model is not None:
        model_class = type(model).__name__
        attention_implementation = getattr(
            model.config, "_attn_implementation", None
        )
        generation_config = _generation_summary(model)
        if generation_config["eos_token_id"] != config["native_eos_token_ids"]:
            raise ContractError("loaded model native EOS configuration changed")
        if generation_config["pad_token_id"] != config["native_pad_token_id"]:
            raise ContractError("loaded model native pad configuration changed")
        parameter_devices = sorted(
            {str(parameter.device) for parameter in model.parameters()}
        )

    completed_model_calls = 0

    def infer(spec):
        nonlocal completed_model_calls
        print(
            f"model attempt {completed_model_calls + 1}/18: "
            f"{spec['record_id']} {spec['mode']}",
            flush=True,
        )
        built = build_model_inputs(
            processor,
            spec,
            image_path=args.image,
            config=config,
        )
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        result = generate_from_audited_inputs(
            model,
            processor,
            spec,
            built,
            audit,
            config,
            inference_context=torch.inference_mode,
            move_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        completed_model_calls += 1
        return result

    attempts = execute_nano_attempt_plan(
        plan,
        infer=infer if model is not None else None,
        model_load_error=model_load_error,
    )
    raw_results = {
        "schema_version": "cosmos3_nano_raw_results.v1",
        "experiment_id": config["experiment_id"],
        "provenance": {
            "freeze_tag": config["freeze_tag"],
            "head_commit": head_commit,
            "git_clean": True,
            "config_sha256": config_sha256,
            "fixture_set_sha256": config["fixture_set_sha256"],
            "prompt_sha256": config["prompt_sha256"],
            "image_sha256": config["image_sha256"],
            "model_view_manifest_sha256": config["model_view_manifest_sha256"],
            "runtime_lock_sha256": config["runtime_lock_sha256"],
            "input_audit_sha256": config["input_audit_sha256"],
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
            "chat_template_file_sha256": config["chat_template_file_sha256"],
            "chat_template_text_sha256": audit["chat_template_text_sha256"],
            "image_preprocessor_sha256": audit["image_preprocessor_sha256"],
            "model_class": model_class,
            "attention_implementation": attention_implementation,
            "generation_config": generation_config,
            "parameter_devices": parameter_devices,
            "model_load_seconds": model_load_seconds,
            "model_load_error": model_load_error,
            "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_name": device_name,
            "compute_capability": compute_capability,
            "cuda_version": torch.version.cuda,
            "driver_version": driver_version,
            "native_eos_token_ids": config["native_eos_token_ids"],
            "native_pad_token_id": config["native_pad_token_id"],
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
        },
        "attempts": attempts,
    }
    output = publish_immutable_json(ROOT, config["output_path"], raw_results)
    completed = sum(row["attempt_status"] == "completed" for row in attempts)
    print(
        f"published {len(attempts)} rows to {output}; "
        f"completed={completed}, errors={len(attempts) - completed}; raw responses not printed"
    )


if __name__ == "__main__":
    try:
        main()
    except (ContractError, FatalRunError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

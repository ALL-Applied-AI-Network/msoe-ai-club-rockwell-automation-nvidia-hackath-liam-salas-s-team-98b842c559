#!/usr/bin/env python3
"""Run the frozen local Cosmos3-Nano stage/vial matrix exactly once."""

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
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.cosmos_contract_runner import (  # noqa: E402
    publish_immutable_json,
    require_output_absent,
    sha256_path,
    verify_git_freeze,
    verify_runtime_environment,
    verify_source_hashes,
)
from fluidmix.explanation.vlm.cosmos_nano_model_view import verify_model_view  # noqa: E402
from fluidmix.explanation.vlm.cosmos_nano_runner import FatalRunError, load_nano_model  # noqa: E402
from fluidmix.explanation.template.explanation_contract import ContractError  # noqa: E402
from fluidmix.explanation.vlm.stage_cosmos_contract import (  # noqa: E402
    SOURCE_PATHS,
    validate_stage_cosmos_config,
)
from fluidmix.explanation.vlm.stage_cosmos_inputs import (  # noqa: E402
    build_stage_model_input_batch,
    verify_tracked_stage_input_audit,
)
from evaluation.scoring.stage_cosmos_plan_artifact import (  # noqa: E402
    load_stage_attempt_plan_artifact,
)
from fluidmix.explanation.vlm.stage_cosmos_runner import (  # noqa: E402
    execute_stage_attempt_plan,
    generate_from_audited_stage_input_batch,
)


PROMPT_PATH = ROOT / "data/cosmos_stage_prompt_v1.txt"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--measured-evidence", type=Path, required=True)
    parser.add_argument("--attempt-plan", type=Path, required=True)
    parser.add_argument("--image-locator", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--input-audit", type=Path, required=True)
    return parser.parse_args()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _driver_version():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    values = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(values) != 1 or not values[0]:
        raise ContractError("could not record one NVIDIA driver version")
    return values[0]


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
    config = _load(args.config)
    validate_stage_cosmos_config(config)
    measured = _load(args.measured_evidence)
    locator = _load(args.image_locator)
    for path, expected, role in (
        (args.measured_evidence, config["measured_evidence_sha256"], "evidence"),
        (args.image_locator, config["image_locator_sha256"], "image locator"),
        (PROMPT_PATH, config["prompt_template_sha256"], "prompt"),
    ):
        if sha256_path(path) != expected:
            raise ContractError(f"stage {role} SHA-256 mismatch")
    if (
        not isinstance(locator, dict)
        or locator.get("schema_version") != 1
        or locator.get("data_origin") != "twin"
        or not isinstance(locator.get("images"), dict)
    ):
        raise ContractError("stage image locator schema is invalid")
    config_sha256 = sha256_path(args.config)
    verify_model_view(args.model_dir, args.model_manifest, config)
    runtime_versions = verify_runtime_environment(args.runtime_lock, config)
    source_hashes = verify_source_hashes(ROOT, config["source_sha256"], SOURCE_PATHS)
    head_commit = verify_git_freeze(ROOT, config["freeze_tag"])
    require_output_absent(ROOT, config["output_path"])

    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != config["runtime_versions"]["transformers"]:
        raise ContractError("imported Transformers version differs from the stage pin")
    attempt_plan_artifact, attempt_plan_sha256 = load_stage_attempt_plan_artifact(
        args.attempt_plan,
        measured,
        PROMPT_PATH.read_text(encoding="utf-8"),
        measured_evidence_sha256=config["measured_evidence_sha256"],
        prompt_template_sha256=config["prompt_template_sha256"],
    )
    plan = attempt_plan_artifact["attempts"]
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    audit = verify_tracked_stage_input_audit(
        args.input_audit,
        plan,
        processor,
        image_locators=locator["images"],
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("stage Nano run requires CUDA with bf16 support")
    device_name = torch.cuda.get_device_name(0)
    compute_capability = list(torch.cuda.get_device_capability(0))
    driver_version = _driver_version()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.reset_peak_memory_stats()
    started_at = utc_now()

    model = None
    model_load_error = None
    load_started = time.monotonic()
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
            raise FatalRunError("fatal memory exhaustion while loading stage Nano") from exc
        model_load_seconds = round(time.monotonic() - load_started, 6)
        model_load_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}

    model_class = attention_implementation = generation_config = None
    parameter_devices = []
    if model is not None:
        model_class = type(model).__name__
        attention_implementation = getattr(model.config, "_attn_implementation", None)
        generation_config = _generation_summary(model)
        if generation_config["eos_token_id"] != config["native_eos_token_ids"]:
            raise ContractError("stage Nano native EOS configuration changed")
        if generation_config["pad_token_id"] != config["native_pad_token_id"]:
            raise ContractError("stage Nano native pad configuration changed")
        parameter_devices = sorted({str(p.device) for p in model.parameters()})

    completed_model_calls = 0

    def infer_batch(specs):
        nonlocal completed_model_calls
        print(
            f"stage model batch {completed_model_calls + 1}-"
            f"{completed_model_calls + len(specs)}/36: "
            f"{specs[0]['condition_id']}",
            flush=True,
        )
        built = build_stage_model_input_batch(
            processor,
            specs,
            image_locators=locator["images"],
            config=config,
        )
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        result = generate_from_audited_stage_input_batch(
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
        completed_model_calls += len(specs)
        return result

    attempts = execute_stage_attempt_plan(
        plan,
        infer_batch=infer_batch if model is not None else None,
        batch_size=config["batch_size"],
        model_load_error=model_load_error,
    )
    raw_results = {
        "schema_version": "stage_cosmos3_nano_raw_results.v1",
        "experiment_id": config["experiment_id"],
        "provenance": {
            "freeze_tag": config["freeze_tag"],
            "head_commit": head_commit,
            "git_clean": True,
            "config_sha256": config_sha256,
            "measured_evidence_sha256": config["measured_evidence_sha256"],
            "image_locator_sha256": config["image_locator_sha256"],
            "prompt_template_sha256": config["prompt_template_sha256"],
            "input_audit_sha256": config["input_audit_sha256"],
            "attempt_plan_sha256": attempt_plan_sha256,
            "source_sha256": source_hashes,
            "model_repository": config["model_repository"],
            "model_revision": config["model_revision"],
            "seed": config["seed"],
            "do_sample": config["do_sample"],
            "max_new_tokens": config["max_new_tokens"],
            "batch_size": config["batch_size"],
            "batch_size_smoke_sha256": config["batch_size_smoke_sha256"],
            "dtype": config["dtype"],
            "python": platform.python_version(),
            "platform_machine": platform.machine(),
            "runtime_versions": runtime_versions,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__,
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
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
        },
        "attempts": attempts,
    }
    output = publish_immutable_json(ROOT, config["output_path"], raw_results)
    completed = sum(row["attempt_status"] == "completed" for row in attempts)
    print(
        f"published 48 stage rows to {output}; completed={completed}, "
        f"errors={48 - completed}; raw responses not printed"
    )


if __name__ == "__main__":
    try:
        main()
    except (ContractError, FatalRunError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

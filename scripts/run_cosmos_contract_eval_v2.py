#!/usr/bin/env python3
"""Run the frozen local Cosmos v2 no-thinking contract matrix exactly once."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_contract_runner import (
    build_attempt_plan,
    create_frozen_output_parent,
    execute_attempt_plan,
    sha256_path,
    verify_git_freeze,
    verify_model_snapshot,
    verify_runtime_environment,
)
from fluidmix.cosmos_input_audit import build_model_messages, verify_tracked_input_audit
from fluidmix.explanation_contract import (
    ContractError,
    validate_eval_config,
    validate_fixture_set,
)

CONFIG_PATH = ROOT / "data/cosmos_contract_eval_v2.json"
FIXTURE_PATH = ROOT / "data/cosmos_contract_fixtures_v1.json"
PROMPT_PATH = ROOT / "data/cosmos_operator_prompt_v1.txt"
AUDIT_PATH = ROOT / "data/cosmos_contract_v2_input_audit.json"


class ModelUnavailable(RuntimeError):
    """The one frozen model load failed, so each planned model attempt records an error."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    validate_eval_config(config)
    validate_fixture_set(fixtures)

    if sha256_path(FIXTURE_PATH) != config["fixture_set_sha256"]:
        raise SystemExit("fixture-set SHA-256 mismatch")
    if sha256_path(PROMPT_PATH) != config["prompt_sha256"]:
        raise SystemExit("prompt SHA-256 mismatch")
    if sha256_path(args.image) != config["image_sha256"]:
        raise SystemExit("fixed image SHA-256 mismatch")
    verify_model_snapshot(args.model_dir, args.model_manifest, config)
    runtime_versions = verify_runtime_environment(args.runtime_lock, config)
    head_commit = verify_git_freeze(ROOT, config["freeze_tag"])

    from PIL import Image

    with Image.open(args.image) as image:
        image.load()
        if image.mode != "RGB" or image.size != (640, 480):
            raise SystemExit("fixed image must be RGB 640x480")

    attempt_plan = build_attempt_plan(fixtures, config, prompt_template)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch as imported_torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != runtime_versions["transformers"]:
        raise ContractError("imported Transformers version does not match verified runtime")

    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    verified_audit_digest = verify_tracked_input_audit(
        AUDIT_PATH,
        attempt_plan,
        processor,
        image_path=args.image,
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    output_path = create_frozen_output_parent(ROOT, config["output_path"])

    started_at = utc_now()
    versions = runtime_versions
    device_name = None
    compute_capability = None
    cuda_version = None
    model_load_seconds = 0.0
    model_load_error = None
    model = None
    torch = imported_torch
    load_started = None

    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ModelUnavailable("CUDA with BF16 support is required")
        device_name = torch.cuda.get_device_name(0)
        compute_capability = list(torch.cuda.get_device_capability(0))
        cuda_version = torch.version.cuda
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        load_started = time.monotonic()
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_dir,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        model.eval()
        torch.cuda.synchronize()
        model_load_seconds = round(time.monotonic() - load_started, 6)
        parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
        if any(not device.startswith("cuda") for device in parameter_devices):
            raise ModelUnavailable(f"model parameters not entirely on CUDA: {parameter_devices}")
    except Exception as exc:
        if load_started is not None:
            model_load_seconds = round(time.monotonic() - load_started, 6)
        model = None
        model_load_error = {"type": type(exc).__name__, "message": str(exc)[:2000]}

    def infer(prompt, image_attached):
        if model is None or processor is None or torch is None:
            detail = model_load_error or {"type": "ModelUnavailable", "message": "unknown"}
            raise ModelUnavailable(f"{detail['type']}: {detail['message']}")
        messages = build_model_messages(prompt, image_attached, args.image)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=config["enable_thinking"],
        ).to(model.device, torch.bfloat16)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=config["do_sample"],
                max_new_tokens=config["max_new_tokens"],
            )
        torch.cuda.synchronize()
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        completion = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return completion, int(trimmed[0].shape[-1])

    attempts = execute_attempt_plan(attempt_plan, infer)
    raw_results = {
        "schema_version": "operator_contract_raw_results.v1",
        "experiment_id": config["experiment_id"],
        "provenance": {
            "freeze_tag": config["freeze_tag"],
            "head_commit": head_commit,
            "git_clean": True,
            "fixture_set_sha256": config["fixture_set_sha256"],
            "prompt_sha256": config["prompt_sha256"],
            "image_sha256": config["image_sha256"],
            "model_manifest_sha256": config["model_manifest_sha256"],
            "runtime_lock_sha256": config["runtime_lock_sha256"],
            "model_revision": config["model_revision"],
            "seed": config["seed"],
            "do_sample": config["do_sample"],
            "max_new_tokens": config["max_new_tokens"],
            "dtype": config["dtype"],
            "python": platform.python_version(),
            "versions": versions,
            "platform_machine": platform.machine(),
            "device_name": device_name,
            "compute_capability": compute_capability,
            "cuda_version": cuda_version,
            "model_load_seconds": model_load_seconds,
            "model_load_error": model_load_error,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "v1_freeze_tag": config["v1_freeze_tag"],
            "enable_thinking": config["enable_thinking"],
            "input_audit_sha256": verified_audit_digest,
            "chat_template_sha256": config["chat_template_sha256"],
        },
        "attempts": attempts,
    }
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(raw_results, stream, indent=2, sort_keys=True)
        stream.write("\n")
    completed = sum(item["attempt_status"] == "completed" for item in attempts)
    print(
        f"wrote {len(attempts)} frozen attempts to {output_path}; "
        f"completed={completed}, errors={len(attempts) - completed}; responses not printed"
    )


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc

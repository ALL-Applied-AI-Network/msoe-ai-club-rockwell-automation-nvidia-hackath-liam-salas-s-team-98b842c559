#!/usr/bin/env python3
"""Create the processor-only audit for the frozen stage/vial Nano matrix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_contract_runner import sha256_path, verify_runtime_environment  # noqa: E402
from fluidmix.cosmos_nano_model_view import verify_model_view  # noqa: E402
from fluidmix.explanation_contract import ContractError  # noqa: E402
from fluidmix.stage_cosmos_contract import (  # noqa: E402
    validate_stage_cosmos_audit_candidate_config,
    validate_stage_cosmos_config,
)
from fluidmix.stage_cosmos_inputs import (  # noqa: E402
    audit_stage_model_inputs,
    stage_audit_json_bytes,
)
from fluidmix.stage_cosmos_plan_artifact import (  # noqa: E402
    load_stage_attempt_plan_artifact,
)


PROMPT_PATH = ROOT / "data/cosmos_stage_prompt_v1.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--measured-evidence", type=Path, required=True)
    parser.add_argument("--attempt-plan", type=Path, required=True)
    parser.add_argument("--image-locator", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    config = _load(args.config)
    candidate = config.get("config_status") == "audit_candidate"
    if candidate:
        validate_stage_cosmos_audit_candidate_config(config)
    else:
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
    attempt_plan_artifact, _ = load_stage_attempt_plan_artifact(
        args.attempt_plan,
        measured,
        PROMPT_PATH.read_text(encoding="utf-8"),
        measured_evidence_sha256=config["measured_evidence_sha256"],
        prompt_template_sha256=config["prompt_template_sha256"],
    )
    if (
        not isinstance(locator, dict)
        or locator.get("schema_version") != 1
        or locator.get("data_origin") != "twin"
        or not isinstance(locator.get("images"), dict)
    ):
        raise ContractError("stage image locator schema is invalid")
    verify_model_view(args.model_dir, args.model_manifest, config)
    verify_runtime_environment(args.runtime_lock, config)
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        raise ContractError("stage input-audit output must be absent")

    import transformers
    from transformers import AutoProcessor

    if transformers.__version__ != config["runtime_versions"]["transformers"]:
        raise ContractError("imported Transformers version differs from the stage pin")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    plan = attempt_plan_artifact["attempts"]
    audit = audit_stage_model_inputs(
        plan,
        processor,
        image_locators=locator["images"],
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    payload = stage_audit_json_bytes(audit)
    actual = hashlib.sha256(payload).hexdigest()
    if not candidate and actual != config["input_audit_sha256"]:
        raise ContractError(
            "computed stage input audit differs from the frozen config hash"
        )
    with args.out.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"wrote 36 stage processor records; sha256={actual}")


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

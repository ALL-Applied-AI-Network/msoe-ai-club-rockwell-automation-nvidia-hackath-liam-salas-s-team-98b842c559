#!/usr/bin/env python3
"""Write the frozen processor-only input audit for Cosmos contract v2."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_contract_runner import (
    build_attempt_plan,
    sha256_path,
    verify_runtime_environment,
)
from fluidmix.cosmos_input_audit import audit_json_bytes, audit_model_inputs
from fluidmix.explanation_contract import (
    ContractError,
    validate_eval_config,
    validate_fixture_set,
)


CONFIG_PATH = ROOT / "data/cosmos_contract_eval_v2.json"
FIXTURE_PATH = ROOT / "data/cosmos_contract_fixtures_v1.json"
PROMPT_PATH = ROOT / "data/cosmos_operator_prompt_v1.txt"
AUDIT_PATH = ROOT / "data/cosmos_contract_v2_input_audit.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    return parser.parse_args()


def verify_processor_assets(model_dir, manifest_path, config):
    """Verify the manifest identity and chat-template bytes without reading weights."""
    model_dir = Path(model_dir)
    if model_dir.is_symlink() or not model_dir.is_dir():
        raise ContractError("model directory must be a real directory")
    model_dir = model_dir.resolve()
    if sha256_path(manifest_path) != config["model_manifest_sha256"]:
        raise ContractError("model manifest SHA-256 mismatch")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("status") != "verified":
        raise ContractError("model manifest is not verified")
    if manifest.get("repo_id") != config["model_repository"]:
        raise ContractError("model repository mismatch")
    if manifest.get("revision") != config["model_revision"]:
        raise ContractError("model revision mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ContractError("model manifest files must be a list")
    template_rows = [row for row in rows if row.get("path") == "chat_template.jinja"]
    if len(template_rows) != 1:
        raise ContractError("model manifest must contain one chat template")
    row = template_rows[0]
    if row.get("sha256") != config["chat_template_sha256"]:
        raise ContractError("manifest chat-template SHA-256 mismatch")
    template_path = model_dir / "chat_template.jinja"
    if sha256_path(template_path) != config["chat_template_sha256"]:
        raise ContractError("chat-template SHA-256 mismatch")
    return model_dir


def main():
    args = parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    validate_eval_config(config)
    validate_fixture_set(fixtures)
    if sha256_path(FIXTURE_PATH) != config["fixture_set_sha256"]:
        raise ContractError("fixture-set SHA-256 mismatch")
    if sha256_path(PROMPT_PATH) != config["prompt_sha256"]:
        raise ContractError("prompt SHA-256 mismatch")
    if sha256_path(args.image) != config["image_sha256"]:
        raise ContractError("fixed image SHA-256 mismatch")
    model_dir = verify_processor_assets(args.model_dir, args.model_manifest, config)
    runtime_versions = verify_runtime_environment(args.runtime_lock, config)
    if AUDIT_PATH.exists() or AUDIT_PATH.is_symlink():
        raise ContractError("tracked input-audit path is already occupied")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import transformers
    from transformers import AutoProcessor

    if transformers.__version__ != runtime_versions["transformers"]:
        raise ContractError("imported Transformers version does not match verified runtime")

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    attempt_plan = build_attempt_plan(fixtures, config, prompt_template)
    audit = audit_model_inputs(
        attempt_plan,
        processor,
        image_path=args.image,
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    payload = audit_json_bytes(audit)
    with AUDIT_PATH.open("xb") as stream:
        stream.write(payload)
    print(
        f"wrote {audit['model_attempt_count']} processor-only input pairs to "
        f"{AUDIT_PATH}; SHA-256={sha256_path(AUDIT_PATH)}"
    )


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc

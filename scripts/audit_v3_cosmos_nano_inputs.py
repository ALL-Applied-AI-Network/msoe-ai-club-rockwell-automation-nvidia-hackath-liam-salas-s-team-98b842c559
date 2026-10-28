#!/usr/bin/env python3
"""Freeze processor tensors for all 54 Varied V3 Nano model attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--attempt-plan", type=Path, required=True)
    parser.add_argument("--image-locator", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--frozen-config-out", type=Path, required=True)
    args = parser.parse_args(argv)

    from transformers import AutoProcessor
    import transformers

    from fluidmix.cosmos_contract_runner import (
        verify_git_freeze,
        verify_runtime_environment,
        verify_source_hashes,
    )
    from fluidmix.cosmos_nano_model_view import verify_model_view
    from fluidmix.v3_cosmos_contract import (
        SOURCE_PATHS,
        validate_v3_cosmos_audit_candidate_config,
        validate_v3_cosmos_config,
    )
    from fluidmix.v3_cosmos_inputs import (
        audit_v3_model_inputs,
        v3_audit_json_bytes,
    )
    from fluidmix.v3_cosmos_plan_artifact import load_v3_attempt_plan_artifact
    from fluidmix.vision_measurement import _publish_json_no_clobber

    candidate = _load(args.candidate_config)
    validate_v3_cosmos_audit_candidate_config(candidate)
    config = dict(candidate)
    config["config_status"] = "execution_frozen"
    config["input_audit_sha256"] = "f" * 64
    validate_v3_cosmos_config(config)
    prompt_path = ROOT / "data/cosmos_f2_varied_prompt_v1.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    for path, expected, role in (
        (args.decisions, config["decision_artifact_sha256"], "decisions"),
        (args.attempt_plan, config["attempt_plan_sha256"], "attempt plan"),
        (args.image_locator, config["image_locator_sha256"], "image locator"),
        (prompt_path, config["prompt_template_sha256"], "prompt"),
    ):
        if _sha(path) != expected:
            raise ValueError(f"V3 {role} SHA-256 mismatch")
    decisions = _load(args.decisions)
    locator = _load(args.image_locator)
    if (
        not isinstance(locator, dict)
        or locator.get("artifact_type") != "f2_varied_image_locators"
        or locator.get("status") != "published"
        or not isinstance(locator.get("images"), dict)
    ):
        raise ValueError("V3 image locator artifact is malformed")
    plan_artifact, _ = load_v3_attempt_plan_artifact(
        args.attempt_plan,
        decisions,
        prompt,
        decision_artifact_sha256=config["decision_artifact_sha256"],
        image_locator_sha256=config["image_locator_sha256"],
    )
    verify_model_view(args.model_dir, args.model_manifest, config)
    verify_runtime_environment(args.runtime_lock, config)
    verify_source_hashes(ROOT, config["source_sha256"], SOURCE_PATHS)
    verify_git_freeze(ROOT, config["freeze_tag"])
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    audit = audit_v3_model_inputs(
        plan_artifact["attempts"],
        processor,
        image_locators=locator["images"],
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    expected_bytes = v3_audit_json_bytes(audit)
    _publish_json_no_clobber(args.audit_out, audit)
    if args.audit_out.read_bytes() != expected_bytes:
        raise ValueError("published V3 audit bytes changed")
    audit_sha = _sha(args.audit_out)
    config["input_audit_sha256"] = audit_sha
    validate_v3_cosmos_config(config)
    _publish_json_no_clobber(args.frozen_config_out, config)
    print(
        f"audited 54 inputs in {audit['batch_count']} batches; "
        f"audit_sha256={audit_sha}; frozen_config={args.frozen_config_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

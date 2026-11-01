#!/usr/bin/env python3
"""Create the frozen processor-only audit for the Nano usable treatment."""

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.cosmos_contract_runner import sha256_path, verify_runtime_environment  # noqa: E402
from fluidmix.explanation.vlm.cosmos_nano_input_audit import audit_json_bytes, audit_model_inputs  # noqa: E402
from fluidmix.explanation.vlm.cosmos_nano_model_view import verify_model_view  # noqa: E402
from fluidmix.explanation.vlm.cosmos_nano_usable_contract import validate_nano_usable_config  # noqa: E402
from fluidmix.explanation.vlm.cosmos_usable_output import build_usable_attempt_plan  # noqa: E402
from fluidmix.explanation.template.explanation_contract import ContractError, validate_fixture_set  # noqa: E402


FIXTURE_PATH = ROOT / "data/cosmos_contract_fixtures_v1.json"
PROMPT_PATH = ROOT / "data/cosmos_operator_prompt_usable_v1.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_nano_usable_config(config)
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    validate_fixture_set(fixtures)
    for path, expected, role in (
        (FIXTURE_PATH, config["fixture_set_sha256"], "fixture"),
        (PROMPT_PATH, config["prompt_sha256"], "prompt"),
        (args.image, config["image_sha256"], "image"),
    ):
        if sha256_path(path) != expected:
            raise ContractError(f"usable {role} SHA-256 mismatch")
    verify_model_view(args.model_dir, args.model_manifest, config)
    verify_runtime_environment(args.runtime_lock, config)
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        raise ContractError("input-audit output must be absent in an existing directory")

    from PIL import Image
    import transformers
    from transformers import AutoProcessor

    if transformers.__version__ != config["runtime_versions"]["transformers"]:
        raise ContractError("imported Transformers version differs from frozen pin")
    with Image.open(args.image) as image:
        image.load()
        if image.mode != "RGB" or image.size != (640, 480):
            raise ContractError("fixed image must be RGB 640x480")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    plan = build_usable_attempt_plan(
        fixtures, config, PROMPT_PATH.read_text(encoding="utf-8")
    )
    audit = audit_model_inputs(
        plan,
        processor,
        image_path=args.image,
        config=config,
        observed_transformers_version=transformers.__version__,
    )
    payload = audit_json_bytes(audit)
    with args.out.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        f"wrote {audit['model_attempt_count']} usable processor records; "
        f"sha256={sha256_path(args.out)}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

#!/usr/bin/env python3
"""Stage the pinned 48-file Nano reasoner subset as an offline hard-link view."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.cosmos_contract_runner import sha256_path  # noqa: E402
from fluidmix.cosmos_nano_model_view import stage_model_view  # noqa: E402
from fluidmix.explanation_contract import ContractError  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = stage_model_view(
        args.snapshot, args.model_dir, args.model_manifest
    )
    print(
        f"staged {manifest['file_count']} regular files "
        f"({manifest['total_bytes']} bytes); "
        f"manifest_sha256={sha256_path(args.model_manifest)}"
    )


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        raise SystemExit(str(exc)) from exc

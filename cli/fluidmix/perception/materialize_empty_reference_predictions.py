#!/usr/bin/env python3
"""Prepare or materialize accepted same-run pre-pour RGB mask predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="project corrected V3 metadata into safe RGB inputs")
    prepare.add_argument("--image-locators", required=True, type=Path)
    prepare.add_argument("--downstream-validation-receipt", required=True, type=Path)
    prepare.add_argument("--request-bundle", required=True, type=Path)
    prepare.add_argument("--recovered-result", required=True, type=Path)
    prepare.add_argument("--independent-review", required=True, type=Path)
    prepare.add_argument("--out", required=True, type=Path)
    materialize = commands.add_parser("materialize", help="publish RGB-only masks and locators")
    materialize.add_argument("--safe-manifest", required=True, type=Path)
    materialize.add_argument("--output-dir", required=True, type=Path)
    materialize.add_argument("--materializer-commit", required=True)
    materialize.add_argument("--cnn-checkpoint", type=Path)
    materialize.add_argument("--cnn-replay", type=Path)
    materialize.add_argument("--cnn-review", type=Path)
    args = parser.parse_args(argv)

    from fluidmix.perception.materializer import (
        materialize_predictions,
        prepare_v3_safe_manifest,
    )

    if args.command == "prepare":
        artifact = prepare_v3_safe_manifest(
            image_locators_path=args.image_locators,
            downstream_validation_receipt_path=args.downstream_validation_receipt,
            request_bundle_path=args.request_bundle,
            recovered_result_path=args.recovered_result,
            independent_review_path=args.independent_review,
            output_path=args.out,
        )
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "images": len(artifact["records"]),
                    "output": os.fspath(args.out.resolve()),
                    "sha256": _sha256(args.out),
                },
                sort_keys=True,
            )
        )
        return 0
    receipt = materialize_predictions(
        safe_manifest_path=args.safe_manifest,
        output_dir=args.output_dir,
        materializer_commit=args.materializer_commit,
        cnn_checkpoint_path=args.cnn_checkpoint,
        cnn_replay_path=args.cnn_replay,
        cnn_review_path=args.cnn_review,
    )
    receipt_path = args.output_dir / "materialization-receipt.json"
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "images": receipt["denominators"]["images"],
                "output": os.fspath(args.output_dir.resolve()),
                "receipt_sha256": _sha256(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

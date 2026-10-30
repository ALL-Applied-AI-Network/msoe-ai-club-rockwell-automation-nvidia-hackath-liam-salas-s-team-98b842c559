#!/usr/bin/env python3
"""Populate a fresh WP5 SQLite store from accepted predicted-mask artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-locators", required=True)
    parser.add_argument("--materialization-receipt", required=True)
    parser.add_argument("--measured-evidence", required=True)
    parser.add_argument("--fill-clip-manifest", required=True)
    parser.add_argument("--downstream-validation-receipt", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--owner-thread-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.predicted_store_bridge import populate_predicted_evidence_store

    receipt = populate_predicted_evidence_store(
        prediction_locators_path=args.prediction_locators,
        materialization_receipt_path=args.materialization_receipt,
        measured_evidence_path=args.measured_evidence,
        fill_clip_manifest_path=args.fill_clip_manifest,
        downstream_validation_receipt_path=args.downstream_validation_receipt,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        owner_thread_id=args.owner_thread_id,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output_dir": str(Path(args.output_dir).expanduser().absolute()),
                "denominators": receipt["denominators"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

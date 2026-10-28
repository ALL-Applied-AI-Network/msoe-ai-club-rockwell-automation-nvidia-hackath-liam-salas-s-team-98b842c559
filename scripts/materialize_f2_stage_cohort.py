#!/usr/bin/env python3
"""Materialize F2 batch publish requests and merge their published contexts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    requests = subparsers.add_parser(
        "requests", help="write one publication request per source-plan batch"
    )
    requests.add_argument("--plan", required=True)
    requests.add_argument("--accepted-qualification-receipt", required=True)
    requests.add_argument("--supervisor-root", required=True)
    requests.add_argument("--runtime-id", required=True)
    requests.add_argument("--output-dir", required=True)

    merge = subparsers.add_parser(
        "merge-contexts", help="merge every request-bound published batch context"
    )
    merge.add_argument("--request-bundle", required=True)
    merge.add_argument("--publication-dir", action="append", required=True)
    merge.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.f2_stage_cohort_materializer import (
        materialize_publish_requests,
        merge_publication_contexts,
    )

    if args.command == "requests":
        receipt = materialize_publish_requests(
            plan_path=args.plan,
            accepted_qualification_receipt=args.accepted_qualification_receipt,
            supervisor_root=args.supervisor_root,
            runtime_id=args.runtime_id,
            output_dir=args.output_dir,
        )
    else:
        receipt = merge_publication_contexts(
            request_bundle=args.request_bundle,
            publication_dirs=args.publication_dir,
            output_dir=args.output_dir,
        )
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

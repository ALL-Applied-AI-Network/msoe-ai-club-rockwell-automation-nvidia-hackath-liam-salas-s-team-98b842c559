#!/usr/bin/env python3
"""Fail closed over the frozen WP1 manifests and post-render evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.wp1_postrender import validate_wp1_acceptance  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path, action="append")
    parser.add_argument(
        "--transport-acceptance", required=True, type=Path, action="append"
    )
    parser.add_argument("--review-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_wp1_acceptance(
            manifest_root=args.manifest_root,
            run_roots=args.run_root,
            transport_receipts=args.transport_acceptance,
            review_receipt=args.review_receipt,
            output_path=args.output,
            repo_root=args.repo_root,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"acceptance validation failed before receipt publication: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

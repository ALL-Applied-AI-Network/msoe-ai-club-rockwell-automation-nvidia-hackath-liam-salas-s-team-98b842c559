#!/usr/bin/env python3
"""Publish the frozen WP1 run-1 DP3 visibility decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.postrender.wp1_postrender import decide_wp1_visibility  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = decide_wp1_visibility(
            manifest_root=args.manifest_root,
            run_root=args.run_root,
            output_path=args.output,
            repo_root=args.repo_root,
        )
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"visibility decision failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

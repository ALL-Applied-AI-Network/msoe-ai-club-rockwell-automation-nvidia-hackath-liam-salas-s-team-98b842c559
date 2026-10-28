#!/usr/bin/env python3
"""Build the exact videos/test WP1 package from a passing acceptance receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.wp1_postrender import package_wp1_test_videos  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = package_wp1_test_videos(
            acceptance_path=args.acceptance, output_root=args.output_root
        )
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"WP1 packaging failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

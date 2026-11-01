#!/usr/bin/env python3
"""Produce accepted INN-67 validation wrappers and a complete render inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.memory.reference_bank.postrender_producer import (  # noqa: E402
    PostrenderProductionError,
    produce_postrender_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = produce_postrender_inputs(
            plan_root=args.plan_root,
            render_root=args.render_root,
            output_dir=args.output_dir,
        )
    except (PostrenderProductionError, OSError) as error:
        print(f"INN-67 post-render production refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

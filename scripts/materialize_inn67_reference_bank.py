#!/usr/bin/env python3
"""Materialize the INN-67 development reference bank from accepted CPU inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.inn67_bank_materializer import (  # noqa: E402
    MaterializationError,
    materialize_reference_bank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--render-inventory", required=True, type=Path)
    parser.add_argument("--predicted-store-export", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = materialize_reference_bank(
            plan_root=args.plan_root,
            amendment_path=args.amendment,
            render_inventory_path=args.render_inventory,
            predicted_store_export_path=args.predicted_store_export,
            output_dir=args.output_dir,
        )
    except (MaterializationError, OSError) as error:
        print(f"INN-67 materialization refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

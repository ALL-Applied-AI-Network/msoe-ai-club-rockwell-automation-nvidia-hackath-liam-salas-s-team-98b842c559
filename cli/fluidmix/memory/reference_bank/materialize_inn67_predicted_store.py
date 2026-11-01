#!/usr/bin/env python3
"""Populate the fresh INN-67 predicted store from accepted development renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.memory.reference_bank.predicted_store_producer import (  # noqa: E402
    PredictedStoreProducerError,
    produce_predicted_store,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--materialization-amendment", required=True, type=Path)
    parser.add_argument("--render-inventory", required=True, type=Path)
    parser.add_argument("--cnn-checkpoint", required=True, type=Path)
    parser.add_argument("--cnn-replay", required=True, type=Path)
    parser.add_argument("--cnn-review", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = produce_predicted_store(
            plan_root=args.plan_root,
            amendment_path=args.amendment,
            materialization_amendment_path=args.materialization_amendment,
            render_inventory_path=args.render_inventory,
            cnn_checkpoint_path=args.cnn_checkpoint,
            cnn_replay_path=args.cnn_replay,
            cnn_review_path=args.cnn_review,
            output_dir=args.output_dir,
        )
    except (PredictedStoreProducerError, OSError) as error:
        print(f"INN-67 predicted-store publication refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

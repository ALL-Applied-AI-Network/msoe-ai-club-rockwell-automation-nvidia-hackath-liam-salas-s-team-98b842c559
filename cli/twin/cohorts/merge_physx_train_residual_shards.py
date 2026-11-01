#!/usr/bin/env python3
"""Merge completed PhysX cohort shards and emit a full validated resume attempt."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import cli.twin.cohorts.run_physx_train_residual_cohort as supervisor


def build_parser():
    parser = supervisor.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--shard-root",
        action="append",
        required=True,
        type=Path,
        help="Copied completed shard root; repeat exactly once per shard.",
    )
    parser.add_argument("--expected-shard-count", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = supervisor.merge_completed_shard_roots(
            args,
            shard_roots=args.shard_root,
            expected_shard_count=args.expected_shard_count,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"shard merge failure: {error}", file=sys.stderr)
        return 2
    print(
        "shard_merge=pass "
        f"shards={receipt['shard_count']} cases={receipt['merged_case_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

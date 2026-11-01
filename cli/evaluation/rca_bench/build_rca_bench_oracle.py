#!/usr/bin/env python3
"""Scorer-only hidden-oracle builder; never imported by the inference runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.rca_bench.rca_bench import write_canonical_json_exclusive
from evaluation.rca_bench.rca_bench_oracle_builder import build_hidden_oracle


def _object(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("--rows must contain a JSON list")
    oracle = build_hidden_oracle(
        _object(args.plan), rows, protocol_sha256=args.protocol_sha256,
        accepted_provenance_by_source_item=_object(args.provenance),
    )
    write_canonical_json_exclusive(args.out, oracle)


if __name__ == "__main__":
    main()

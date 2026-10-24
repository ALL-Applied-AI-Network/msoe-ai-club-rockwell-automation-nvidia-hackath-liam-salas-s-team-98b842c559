#!/usr/bin/env python3
"""Score one frozen Cosmos contract result matrix without output repair."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation_scoring import score_raw_results


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.raw_results, args.config, args.oracle):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    if not args.out.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {args.out.parent}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if sha256(args.oracle) != config["oracle_sha256"]:
        raise SystemExit("oracle SHA-256 does not match frozen config")
    raw_results = json.loads(args.raw_results.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    score = score_raw_results(raw_results, config, oracle)
    score["input_sha256"] = {
        "raw_results": sha256(args.raw_results),
        "config": sha256(args.config),
        "oracle": sha256(args.oracle),
    }
    score["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(score, stream, indent=2, sort_keys=True)
        stream.write("\n")
    matrix = score["matrix"]
    policy = score["policy"]
    print(
        f"scored {matrix['attempted']}/{matrix['planned']} attempts; "
        f"policy {policy['passed']}/{policy['total']}; wrote {args.out}"
    )


if __name__ == "__main__":
    main()

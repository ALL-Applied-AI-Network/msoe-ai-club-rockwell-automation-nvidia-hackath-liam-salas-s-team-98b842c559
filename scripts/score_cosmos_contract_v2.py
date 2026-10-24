#!/usr/bin/env python3
"""Score one frozen Cosmos v2 matrix with source provenance and no output repair."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "data/cosmos_contract_v2_input_audit.json"
STRICT_SCORER_PATH = ROOT / "src/fluidmix/explanation_scoring.py"
V2_SCORER_PATH = ROOT / "src/fluidmix/cosmos_v2_scoring.py"
CONTRACT_PATH = ROOT / "src/fluidmix/explanation_contract.py"


def sha256_regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"expected regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.raw_results, args.config, args.oracle):
        sha256_regular(path)
    if not args.out.parent.is_dir():
        raise SystemExit(f"output parent does not exist: {args.out.parent}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_paths = {
        "input_audit_sha256": AUDIT_PATH,
        "scoring_module_sha256": STRICT_SCORER_PATH,
        "scoring_script_sha256": Path(__file__),
        "v2_scoring_module_sha256": V2_SCORER_PATH,
        "contract_module_sha256": CONTRACT_PATH,
    }
    for field, path in source_paths.items():
        if sha256_regular(path) != config.get(field):
            raise SystemExit(f"frozen source SHA-256 mismatch: {field}")
    if sha256_regular(args.oracle) != config.get("oracle_sha256"):
        raise SystemExit("oracle SHA-256 does not match frozen config")

    sys.path.insert(0, str(ROOT / "src"))
    from fluidmix.cosmos_v2_scoring import score_v2_raw_results

    raw_results = json.loads(args.raw_results.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    score = score_v2_raw_results(raw_results, config, oracle)
    score["input_sha256"] = {
        "raw_results": sha256_regular(args.raw_results),
        "config": sha256_regular(args.config),
        "oracle": sha256_regular(args.oracle),
    }
    score["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(score, stream, indent=2, sort_keys=True)
        stream.write("\n")
    matrix = score["matrix"]
    cap = score["generation_cap_reaches"]
    print(
        f"scored {matrix['attempted']}/{matrix['planned']} attempts; "
        f"cap {cap['reached']}/{cap['total']}; wrote {args.out}"
    )


if __name__ == "__main__":
    main()

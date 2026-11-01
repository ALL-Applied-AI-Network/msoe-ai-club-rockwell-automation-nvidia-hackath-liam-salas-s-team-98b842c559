#!/usr/bin/env python3
"""Score a complete Nano matched artifact with the sealed policy data."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.cosmos_contract_runner import (  # noqa: E402
    sha256_path,
    verify_source_hashes,
)
from fluidmix.explanation.vlm.cosmos_nano_contract import (  # noqa: E402
    SOURCE_PATHS,
    validate_nano_eval_config,
)
from fluidmix.explanation.vlm.cosmos_nano_model_view import verify_model_view  # noqa: E402
from evaluation.scoring.cosmos_nano_scoring import score_nano_raw_results  # noqa: E402
from fluidmix.explanation.template.explanation_contract import ContractError  # noqa: E402


ORACLE_SHA256 = "cb2d45ca1fce48e7603a6cde52ff0cbb1bdad11e7b2285ad01b80cbd232faab9"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _publish_score(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ContractError("score output must be absent in the completed run directory")
    payload = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.incomplete"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        path.chmod(0o444)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise


def main():
    args = parse_args()
    if args.raw_results.is_symlink() or not args.raw_results.is_file():
        raise ContractError("raw-results artifact must be a regular file")
    if args.raw_results.stat().st_mode & 0o222:
        raise ContractError("raw-results artifact must be immutable before scoring")
    if args.out.parent.resolve() != args.raw_results.parent.resolve():
        raise ContractError("score output must share the frozen raw-results directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_nano_eval_config(config)
    verify_source_hashes(ROOT, config["source_sha256"], SOURCE_PATHS)
    verify_model_view(args.model_dir, args.model_manifest, config)
    if sha256_path(args.oracle) != ORACLE_SHA256:
        raise ContractError("sealed scoring data SHA-256 mismatch")
    raw_results = json.loads(args.raw_results.read_text(encoding="utf-8"))
    config_sha256 = sha256_path(args.config)
    if raw_results.get("provenance", {}).get("config_sha256") != config_sha256:
        raise ContractError("raw artifact is bound to a different frozen config")
    sealed = json.loads(args.oracle.read_text(encoding="utf-8"))
    score = score_nano_raw_results(raw_results, config, sealed)
    score["input_sha256"] = {
        "raw_results": sha256_path(args.raw_results),
        "config": config_sha256,
        "sealed_scoring_data": sha256_path(args.oracle),
        "model_view_manifest": sha256_path(args.model_manifest),
    }
    score["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    score["historical_edge_v2_contract_only"] = {
        "model_policy_passed": 0,
        "model_policy_total": 18,
        "adapter_and_runtime_matched": False,
    }
    score["claim_boundary"] = (
        "Synthetic interface compliance only; this does not establish visual understanding, "
        "defect or cause accuracy, physical validity, real-camera robustness, or utility over "
        "the deterministic template."
    )
    _publish_score(args.out, score)
    model = score["model_policy"]
    control = score["template_control"]
    print(
        f"scored 24/24 rows; model policy {model['passed']}/{model['total']}; "
        f"template {control['passed']}/{control['total']}; wrote {args.out}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

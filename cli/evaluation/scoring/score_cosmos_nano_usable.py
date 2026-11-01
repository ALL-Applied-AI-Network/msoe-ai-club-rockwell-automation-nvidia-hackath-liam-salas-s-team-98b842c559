#!/usr/bin/env python3
"""Score the frozen Nano usable-output artifact without repairing raw text."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.cosmos_contract_runner import sha256_path, verify_source_hashes  # noqa: E402
from fluidmix.explanation.vlm.cosmos_nano_model_view import verify_model_view  # noqa: E402
from fluidmix.explanation.vlm.cosmos_nano_usable_contract import (  # noqa: E402
    SOURCE_PATHS,
    validate_nano_usable_config,
)
from evaluation.scoring.cosmos_usable_scoring import score_nano_usable_raw_results  # noqa: E402
from fluidmix.explanation.template.explanation_contract import ContractError, validate_fixture_set  # noqa: E402


FIXTURE_PATH = ROOT / "data/cosmos_contract_fixtures_v1.json"
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
        raise ContractError("usable score output must be absent in the run directory")
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
        raise ContractError("usable raw-results artifact must be a regular file")
    if args.raw_results.stat().st_mode & 0o222:
        raise ContractError("usable raw-results artifact must be immutable before scoring")
    if args.out.parent.resolve() != args.raw_results.parent.resolve():
        raise ContractError("usable score must share the raw-results directory")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_nano_usable_config(config)
    verify_source_hashes(ROOT, config["source_sha256"], SOURCE_PATHS)
    verify_model_view(args.model_dir, args.model_manifest, config)
    if sha256_path(FIXTURE_PATH) != config["fixture_set_sha256"]:
        raise ContractError("usable fixture SHA-256 mismatch")
    if sha256_path(args.oracle) != ORACLE_SHA256:
        raise ContractError("sealed scoring data SHA-256 mismatch")
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    validate_fixture_set(fixtures)
    oracle = json.loads(args.oracle.read_text(encoding="utf-8"))
    raw = json.loads(args.raw_results.read_text(encoding="utf-8"))
    config_sha256 = sha256_path(args.config)
    if raw.get("provenance", {}).get("config_sha256") != config_sha256:
        raise ContractError("usable raw artifact is bound to a different config")
    score = score_nano_usable_raw_results(raw, config, fixtures, oracle)
    score["input_sha256"] = {
        "raw_results": sha256_path(args.raw_results),
        "config": config_sha256,
        "fixtures": sha256_path(FIXTURE_PATH),
        "sealed_scoring_data": sha256_path(args.oracle),
        "model_view_manifest": sha256_path(args.model_manifest),
    }
    score["scored_at_utc"] = datetime.now(timezone.utc).isoformat()
    score["claim_boundary"] = (
        "Synthetic explanation-interface compliance only. Lenient parsing never counts as "
        "raw strict-schema success and establishes no detection, cause, physical, or real-camera claim."
    )
    _publish_score(args.out, score)
    print(
        "scored usable 24/24 rows; "
        f"raw strict {score['raw_strict_schema_model']['passed']}/18; "
        f"lenient parse {score['lenient_parse_model']['passed']}/18; "
        f"content {score['model_content_policy']['passed']}/18; "
        f"template {score['template_control']['passed']}/6"
    )


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

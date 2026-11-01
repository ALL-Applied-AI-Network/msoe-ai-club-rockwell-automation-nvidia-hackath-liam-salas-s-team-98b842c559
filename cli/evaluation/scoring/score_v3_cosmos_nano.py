#!/usr/bin/env python3
"""Score V3 Nano raw format, lenient parse, and content separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from fluidmix.explanation.vlm.cosmos_contract import validate_v3_cosmos_config
    from evaluation.scoring.cosmos_scoring import score_v3_cosmos_attempts
    from fluidmix.measurement.vision_measurement import _publish_json_no_clobber

    config = _load(args.config)
    validate_v3_cosmos_config(config)
    decisions = _load(args.decisions)
    raw = _load(args.raw_results)
    if (
        raw.get("schema_version") != "f2_varied_cosmos3_nano_raw_results.v1"
        or raw.get("experiment_id") != config["experiment_id"]
        or not isinstance(raw.get("provenance"), dict)
        or not isinstance(raw.get("attempts"), list)
    ):
        raise ValueError("V3 raw-results artifact is malformed")
    provenance = raw["provenance"]
    expected = {
        "config_sha256": _sha(args.config),
        "decision_artifact_sha256": config["decision_artifact_sha256"],
        "attempt_plan_sha256": config["attempt_plan_sha256"],
        "image_locator_sha256": config["image_locator_sha256"],
        "prompt_template_sha256": config["prompt_template_sha256"],
        "input_audit_sha256": config["input_audit_sha256"],
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("V3 raw-results provenance drifted")
    prompt = (ROOT / "data/cosmos_f2_varied_prompt_v1.txt").read_text(
        encoding="utf-8"
    )
    score = score_v3_cosmos_attempts(raw["attempts"], decisions, prompt)
    artifact = {
        "schema_version": 1,
        "artifact_type": "f2_varied_cosmos_nano_score",
        "status": "scored",
        "raw_results": {"path": str(args.raw_results), "sha256": _sha(args.raw_results)},
        "config_sha256": _sha(args.config),
        "decision_artifact_sha256": config["decision_artifact_sha256"],
        "score": score,
    }
    _publish_json_no_clobber(args.out, artifact)
    metrics = score["metrics"]
    print(
        "scored 72 outcomes: raw strict model "
        f"{metrics['raw_strict_schema_model']['passed']}/54; lenient model "
        f"{metrics['lenient_parse_model']['passed']}/54; content model "
        f"{metrics['strict_content_policy_model']['passed']}/54; "
        f"output_sha256={_sha(args.out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score the immutable stage/vial Nano raw artifact without repairing text."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.cosmos_contract_runner import sha256_path  # noqa: E402
from fluidmix.explanation.template.explanation_contract import ContractError  # noqa: E402
from twin.render.render_process import atomic_write_json  # noqa: E402
from fluidmix.explanation.vlm.stage_cosmos_contract import validate_stage_cosmos_config  # noqa: E402
from evaluation.scoring.stage_cosmos_scoring import score_stage_attempts  # noqa: E402
from evaluation.scoring.stage_cosmos_plan_artifact import (  # noqa: E402
    load_stage_attempt_plan_artifact,
)


PROMPT_PATH = ROOT / "data/cosmos_stage_prompt_v1.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--measured-evidence", type=Path, required=True)
    parser.add_argument("--attempt-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    config = _load(args.config)
    validate_stage_cosmos_config(config)
    raw = _load(args.raw)
    measured = _load(args.measured_evidence)
    if sha256_path(args.measured_evidence) != config["measured_evidence_sha256"]:
        raise ContractError("stage scoring evidence SHA-256 mismatch")
    if sha256_path(PROMPT_PATH) != config["prompt_template_sha256"]:
        raise ContractError("stage scoring prompt SHA-256 mismatch")
    _, attempt_plan_sha256 = load_stage_attempt_plan_artifact(
        args.attempt_plan,
        measured,
        PROMPT_PATH.read_text(encoding="utf-8"),
        measured_evidence_sha256=config["measured_evidence_sha256"],
        prompt_template_sha256=config["prompt_template_sha256"],
    )
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema_version", "experiment_id", "provenance", "attempts"}
        or raw.get("schema_version") != "stage_cosmos3_nano_raw_results.v1"
        or raw.get("experiment_id") != config["experiment_id"]
        or not isinstance(raw.get("provenance"), dict)
        or raw["provenance"].get("config_sha256") != sha256_path(args.config)
        or raw["provenance"].get("attempt_plan_sha256") != attempt_plan_sha256
    ):
        raise ContractError("stage raw-results identity or config binding is invalid")
    for field in (
        "measured_evidence_sha256",
        "image_locator_sha256",
        "prompt_template_sha256",
        "input_audit_sha256",
    ):
        if raw["provenance"].get(field) != config[field]:
            raise ContractError(f"stage raw-results provenance drifted: {field}")
    if os.path.lexists(args.out) or args.out.parent.resolve() != args.raw.parent.resolve():
        raise ContractError("stage score must be absent beside the raw artifact")
    score = score_stage_attempts(
        raw["attempts"], measured, PROMPT_PATH.read_text(encoding="utf-8")
    )
    score.update(
        {
            "experiment_id": config["experiment_id"],
            "raw_results_sha256": sha256_path(args.raw),
            "measured_evidence_sha256": config["measured_evidence_sha256"],
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
            "claim_boundary": (
                "Synthetic twin explanation behavior only; no physical carryover, "
                "defect, cause, plant-action, or sim-to-real claim. Legacy raw JSON, "
                "exact three-line format, lenient parse, and content are separate metrics."
            ),
        }
    )
    atomic_write_json(args.out.parent, args.out.name, score)
    metrics = score["metrics"]
    print(
        "scored 48/48 stage rows; model legacy raw strict "
        f"{metrics['legacy_raw_strict_schema_model']['passed']}/36; "
        f"lenient {metrics['lenient_parse_model']['passed']}/36; "
        f"content {metrics['strict_content_policy_model']['passed']}/36; "
        f"template {metrics['template_control_content_policy']['passed']}/12"
    )


if __name__ == "__main__":
    try:
        main()
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

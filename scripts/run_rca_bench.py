#!/usr/bin/env python3
"""Run the CPU-only RCA-bench control-plane and sealed template phases.

The CLI is deliberately a thin adapter around :mod:`fluidmix.rca_bench_controller`.
``run-template`` does not accept an oracle path; ``score`` is the first phase that
opens the oracle.  No subcommand loads model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_map(path: str | Path) -> dict[str, Any]:
    value = _json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _canonical_json_map(path: str | Path) -> dict[str, Any]:
    from fluidmix.rca_bench import canonical_bytes

    source = Path(path)
    value = _json_map(source)
    if source.read_bytes() != canonical_bytes(value):
        raise ValueError(f"expected canonical JSON artifact bytes: {path}")
    return value


def _common_provenance(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provenance", required=True, help="accepted provenance mapping JSON")


def _freeze_sources(parser: argparse.ArgumentParser) -> None:
    for name in (
        "protocol", "builder-source", "scorer-source", "evidence-schema", "prompt",
        "guard-config", "split-spec", "recipes", "transitions", "real-reserved",
        "manifests", "checker-code",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--test-source", action="append", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--generation-config", required=True)


def _freeze_source_map(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": args.protocol,
        "builder_source": args.builder_source,
        "scorer_source": args.scorer_source,
        "tests": args.test_source,
        "evidence_schema": args.evidence_schema,
        "prompt": args.prompt,
        "guard_config": args.guard_config,
        "partitions": {
            name: getattr(args, name)
            for name in (
                "split_spec", "recipes", "transitions", "real_reserved", "manifests",
                "checker_code",
            )
        },
        "git_commit": args.git_commit,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("publish-public", help="atomically publish a plan/locator pair")
    plan.add_argument("--items", required=True)
    plan.add_argument("--source-artifacts", required=True)
    _common_provenance(plan)
    plan.add_argument("--benchmark-role", required=True)
    plan.add_argument("--development-partition")
    plan.add_argument("--split-version", required=True)
    for name in (
        "prompt-template-sha256", "guard-config-sha256", "evidence-schema-sha256",
        "model-input-schema-sha256",
    ):
        plan.add_argument(f"--{name}", required=True)
    plan.add_argument("--rgb-map", required=True, help="image ID to RGB path JSON object")
    plan.add_argument("--plan-out", required=True)
    plan.add_argument("--locators-out", required=True)

    oracle = sub.add_parser("publish-oracle", help="validate and publish an exact-plan oracle")
    oracle.add_argument("--oracle", required=True)
    oracle.add_argument("--plan", required=True)
    _common_provenance(oracle)
    oracle.add_argument("--out", required=True)

    freeze = sub.add_parser("publish-freeze", help="publish a digest-only freeze receipt")
    freeze.add_argument("--receipt", required=True)
    freeze.add_argument("--plan", required=True)
    freeze.add_argument("--locators", required=True)
    freeze.add_argument("--oracle", required=True)
    _common_provenance(freeze)
    _freeze_sources(freeze)
    freeze.add_argument("--out", required=True)

    template = sub.add_parser("run-template", help="run the deterministic CPU template")
    template.add_argument("--plan", required=True)
    template.add_argument("--locators", required=True)
    template.add_argument("--freeze", required=True)
    _common_provenance(template)
    _freeze_sources(template)
    template.add_argument("--ledger", required=True)
    template.add_argument("--raw", required=True)
    template.add_argument("--fatal-before-attempt", type=int)

    fatal = sub.add_parser("finalize-fatal", help="reopen a fatal journal using sidecars only")
    fatal.add_argument("--ledger", required=True)
    fatal.add_argument("--plan", required=True)
    fatal.add_argument("--raw-bindings", required=True)
    fatal.add_argument("--fatal-event-sha256", required=True)
    fatal.add_argument("--raw", required=True)

    score = sub.add_parser("score", help="reconstruct and validate a sealed score artifact")
    score.add_argument("--plan", required=True)
    score.add_argument("--locators", required=True)
    score.add_argument("--oracle", required=True)
    score.add_argument("--raw", required=True)
    score.add_argument("--freeze", required=True)
    _common_provenance(score)
    score.add_argument("--ledger", required=True, help="canonical JSONL ledger")
    score.add_argument("--expected-bindings", required=True)
    score.add_argument("--method-id", required=True)
    score.add_argument("--out", required=True)
    return parser


def _ledger_events(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "finalize-fatal":
        from fluidmix.rca_bench_fatal import finalize_fatal_reopen

        finalize_fatal_reopen(
            args.ledger, items=_json_map(args.plan)["items"],
            raw_bindings=_json_map(args.raw_bindings),
            fatal_event_sha256=args.fatal_event_sha256, raw_path=args.raw,
        )
        return 0

    from fluidmix.rca_bench import write_canonical_json_exclusive
    from fluidmix.rca_bench_controller import (
        build_image_locators,
        build_public_plan,
        build_score_artifact,
        publish_full_freeze,
        publish_oracle_artifact,
        publish_public_artifacts,
        run_template_baseline,
    )

    if args.command == "publish-public":
        plan = build_public_plan(
            _json(args.items), benchmark_role=args.benchmark_role,
            development_partition=args.development_partition,
            split_version=args.split_version,
            prompt_template_sha256=args.prompt_template_sha256,
            guard_config_sha256=args.guard_config_sha256,
            evidence_schema_sha256=args.evidence_schema_sha256,
            model_input_schema_sha256=args.model_input_schema_sha256,
            source_artifacts=_json(args.source_artifacts),
            accepted_provenance_by_source_item=_json_map(args.provenance),
        )
        locators = build_image_locators(
            plan, rgb_paths=_json_map(args.rgb_map), verify_files=True,
        )
        publish_public_artifacts(
            plan, locators, plan_path=args.plan_out, locator_path=args.locators_out,
            accepted_provenance_by_source_item=_json_map(args.provenance),
        )
    elif args.command == "publish-oracle":
        publish_oracle_artifact(
            _json_map(args.oracle), _json_map(args.plan), path=args.out,
            accepted_provenance_by_source_item=_json_map(args.provenance),
        )
    elif args.command == "publish-freeze":
        publish_full_freeze(
            _canonical_json_map(args.receipt), _canonical_json_map(args.plan),
            _canonical_json_map(args.locators), _canonical_json_map(args.oracle), path=args.out,
            accepted_provenance_by_source_item=_json_map(args.provenance),
            freeze_source_paths=_freeze_source_map(args),
            generation_config_path=args.generation_config,
        )
    elif args.command == "run-template":
        run_template_baseline(
            _json_map(args.plan), _json_map(args.locators), _json_map(args.freeze),
            accepted_provenance_by_source_item=_json_map(args.provenance),
            freeze_source_paths=_freeze_source_map(args),
            generation_config_path=args.generation_config,
            ledger_path=args.ledger, raw_path=args.raw,
            fatal_before_attempt=args.fatal_before_attempt,
        )
    elif args.command == "score":
        score = build_score_artifact(
            _json_map(args.plan), _json_map(args.locators), _json_map(args.oracle),
            _json_map(args.raw), _json_map(args.freeze),
            accepted_provenance_by_source_item=_json_map(args.provenance),
            ledger_events=_ledger_events(args.ledger),
            expected_bindings=_json_map(args.expected_bindings), method_id=args.method_id,
        )
        write_canonical_json_exclusive(args.out, score)
    else:  # pragma: no cover - argparse enforces the choices
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build, validate, publish, or freeze RCA-bench public inputs (CPU only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fluidmix.rca_bench import (
    canonical_bytes,
    canonical_sha256,
    sha256_file,
    validate_image_locators,
    validate_plan,
    write_canonical_json_exclusive,
)
from fluidmix.rca_bench_controller import compute_actual_freeze_bindings
from fluidmix.rca_bench_input_bundle import build_model_freeze_receipt
from fluidmix.rca_bench_input_bundle import (
    build_development_partition_artifacts,
    build_development_public_plans,
    build_test_public_plan,
    build_verified_image_locators,
    project_primary_item,
    publish_development_input_bundles,
    publish_plan_input_bundle,
    validate_cross_role_isolation,
)


def _object(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate public plan and locators")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--locators", required=True)
    validate.add_argument("--provenance", required=True)
    build_test = commands.add_parser(
        "build-test", help="project accepted WP1 records and publish public controller inputs"
    )
    build_test.add_argument("--request", required=True)
    build_test.add_argument("--out-dir", required=True)
    build_development = commands.add_parser(
        "build-development",
        help="project accepted development records, controls, and the sealed 12-pair qualification",
    )
    build_development.add_argument("--request", required=True)
    build_development.add_argument("--out-dir", required=True)
    freeze = commands.add_parser("freeze", help="build one digest-only per-model freeze")
    for flag in ("plan", "locators", "provenance", "out"):
        freeze.add_argument(f"--{flag.replace('_', '-')}", required=True)
    freeze.add_argument("--model-repository", required=True)
    freeze.add_argument("--model-revision", required=True)
    freeze.add_argument("--model-weight-sha256", required=True)
    freeze.add_argument("--oracle-sha256", required=True, help="digest only; no oracle path is accepted")
    freeze.add_argument("--split-id", action="append", required=True)
    freeze.add_argument("--git-commit", required=True)
    for flag in (
        "protocol", "builder-source", "scorer-source", "evidence-schema", "prompt",
        "guard-config", "split-spec", "recipes", "transitions", "real-reserved",
        "manifests", "checker-code", "generation-config",
        "development-partition-receipt", "qualification-access-log",
    ):
        freeze.add_argument(f"--{flag}", required=True)
    freeze.add_argument("--test-source", action="append", required=True)
    return parser


def _actual_freeze_hashes(args: argparse.Namespace) -> tuple[dict, list[str]]:
    generation_path = Path(args.generation_config)
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if not isinstance(generation, dict) or generation_path.read_bytes() != canonical_bytes(generation):
        raise ValueError("generation config must be one canonical JSON object")
    partition_paths = {
        name: getattr(args, name)
        for name in ("split_spec", "recipes", "transitions", "real_reserved", "manifests", "checker_code")
    }
    actual = compute_actual_freeze_bindings(
        protocol_path=args.protocol, builder_source_path=args.builder_source,
        scorer_source_path=args.scorer_source, test_paths=args.test_source,
        evidence_schema_path=args.evidence_schema, prompt_path=args.prompt,
        guard_config_path=args.guard_config, partition_paths=partition_paths,
        git_commit=args.git_commit, generation_config=generation,
    )
    hashes = {
        "protocol": actual["protocol_sha256"],
        "builder_source": actual["builder_source_sha256"],
        "scorer_source": actual["scorer_source_sha256"],
        "evidence_schema": actual["evidence_schema_sha256"],
        "prompt": actual["prompt_sha256"],
        "guard_config": actual["guard_config_sha256"],
        "development_partition_receipt": sha256_file(args.development_partition_receipt),
        "qualification_access_log": sha256_file(args.qualification_access_log),
        "generation_config": canonical_sha256(generation),
        "oracle": args.oracle_sha256,
        **actual["partition_checker_bindings"],
    }
    return hashes, actual["test_hashes"]


def _project_request(request: dict, *, development_partition: str | None) -> tuple[list[dict], dict]:
    projections = request.get("projections")
    if not isinstance(projections, list) or not projections:
        raise ValueError("build request projections must be a nonempty list")
    items = []
    provenance = {}
    for row in projections:
        if not isinstance(row, dict) or set(row) != {
            "item_id", "records", "projection_spec",
            "production_sequence_by_evidence_id", "accepted_provenance",
        }:
            raise ValueError("build request projection fields are invalid")
        item = project_primary_item(
            row["records"], row["projection_spec"],
            row["production_sequence_by_evidence_id"], row["accepted_provenance"],
            item_id=row["item_id"], development_partition=development_partition,
            prompt_template=request["prompt_template"],
        )
        items.append(item)
        provenance[item["source_item_id"]] = row["accepted_provenance"]
    return items, provenance


def main() -> None:
    args = _parser().parse_args()
    if args.command in {"build-test", "build-development"}:
        request = _object(args.request)
        if request.get("schema_version") != "rca-bench-public-build-request-v1":
            raise ValueError("build request schema is invalid")
        expected_request_fields = {
            "schema_version", "prompt_template", "projections", "metadata",
            "partition_leakage_audit",
        } | ({"wp1_source_provenance", "development_publication"}
             if args.command == "build-test" else set())
        if set(request) != expected_request_fields:
            raise ValueError("build request fields are invalid")
        items, provenance = _project_request(
            request, development_partition=None if args.command == "build-test" else "selection"
        )
        if args.command == "build-development":
            partitioned = build_development_partition_artifacts(items, provenance)
            plans = build_development_public_plans(
                partitioned, provenance, metadata=request["metadata"],
                partition_leakage_audit=request["partition_leakage_audit"],
            )
            publish_development_input_bundles(plans, partitioned, output_dir=args.out_dir)
            return
        plan = build_test_public_plan(
            items, provenance, metadata=request["metadata"],
            wp1_source_provenance=request["wp1_source_provenance"],
            partition_leakage_audit=request["partition_leakage_audit"],
        )
        development = request.get("development_publication")
        required_development = {
            "selection_plan", "selection_provenance", "qualification_plan",
            "qualification_provenance", "partition_receipt",
            "publication_receipt", "partition_leakage_audit",
        }
        if not isinstance(development, dict) or set(development) != required_development:
            raise ValueError("test build requires the exact frozen development publication")
        isolation = validate_cross_role_isolation(
            [development["selection_plan"], development["qualification_plan"]], plan,
            development_provenance_by_partition={
                "selection": development["selection_provenance"],
                "grounding_qualification": development["qualification_provenance"],
            },
            test_provenance=provenance,
            development_partition_receipt=development["partition_receipt"],
            development_publication_receipt=development["publication_receipt"],
            development_partition_leakage_audit=development["partition_leakage_audit"],
            test_partition_leakage_audit=request["partition_leakage_audit"],
            wp1_source_provenance=request["wp1_source_provenance"],
        )
        locators = build_verified_image_locators(plan, provenance)
        publish_plan_input_bundle(
            plan, locators, provenance, output_dir=args.out_dir,
            extra_artifacts={
                "rca-bench-cross-role-isolation-v1.json": isolation,
                "rca-bench-wp1-source-provenance-v1.json": request["wp1_source_provenance"],
            },
        )
        return
    plan = _object(args.plan)
    locators = _object(args.locators)
    provenance = _object(args.provenance)
    if args.command == "validate":
        validate_plan(plan, locators=locators, accepted_provenance_by_source_item=provenance)
        validate_image_locators(locators, verify_files=True)
        return
    validate_image_locators(locators, verify_files=True)
    hashes, test_hashes = _actual_freeze_hashes(args)
    receipt = build_model_freeze_receipt(
        plan, locators, model_repository=args.model_repository,
        model_revision=args.model_revision, model_weight_sha256=args.model_weight_sha256,
        oracle_sha256=args.oracle_sha256, split_ids=args.split_id,
        test_hashes=test_hashes, git_commit=args.git_commit,
        hashes=hashes, accepted_provenance_by_source_item=provenance,
    )
    write_canonical_json_exclusive(args.out, receipt)


if __name__ == "__main__":
    main()

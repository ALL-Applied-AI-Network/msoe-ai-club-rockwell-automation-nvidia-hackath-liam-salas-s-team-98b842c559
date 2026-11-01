#!/usr/bin/env python3
"""Build source-only WP1 DP3 fallback plans or register completed cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.postrender.wp1_dp3_fallback import (  # noqa: E402
    TARGETS,
    build_dp3_fallback_plan_artifacts,
    canonical_sha256,
)
from twin.postrender.wp1_postrender import (  # noqa: E402
    _load_frozen_bundle,
    register_wp1_dp3_fallback_case,
)


def _write_new(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="write the two unrendered one-vial plans")
    plan.add_argument("--manifest-root", required=True, type=Path)
    plan.add_argument("--output-root", required=True, type=Path)
    plan.add_argument("--repo-root", type=Path, default=ROOT)
    register = subparsers.add_parser(
        "register", help="derive one fallback acceptance from completed real artifacts"
    )
    register.add_argument("--target-ordinal", required=True, type=int, choices=(5, 9))
    register.add_argument("--manifest-root", required=True, type=Path)
    register.add_argument("--repo-root", type=Path, default=ROOT)
    register.add_argument("--plan-root", required=True, type=Path)
    register.add_argument("--case-root", required=True, type=Path)
    register.add_argument("--host-profile", required=True, type=Path)
    register.add_argument("--offline-validation", required=True, type=Path)
    return parser


def _plan(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_root.resolve()
    if not output.is_dir() or output.is_symlink():
        raise ValueError("output root must be one existing real directory")
    artifacts = _load_frozen_bundle(
        args.manifest_root.resolve(), args.repo_root.resolve()
    )
    collection = artifacts["full-transition-manifest.json"]
    run1 = collection["run_bindings"][0]
    frozen_plan = artifacts[run1["render_plan_path"]]
    rows = []
    for ordinal in sorted(TARGETS):
        target_root = output / f"ordinal-{ordinal:02d}"
        target_root.mkdir(mode=0o750)
        values = build_dp3_fallback_plan_artifacts(
            frozen_plan, target_ordinal=ordinal
        )
        for name, value in values.items():
            _write_new(target_root / name, value)
        plan = values["render-schedule-plan.json"]
        case_id = plan["execution_schedules"][0]["case_binding"]["case_id"]
        rows.append(
            {
                "target_vial_ordinal": ordinal,
                "case_id": case_id,
                "plan_root": str(target_root),
                "render_plan_canonical_sha256": canonical_sha256(plan),
                "render_plan_file_sha256": hashlib.sha256(
                    (target_root / "render-schedule-plan.json").read_bytes()
                ).hexdigest(),
                "render_seed": TARGETS[ordinal]["render_seed"],
                "nuisance_group_id": TARGETS[ordinal]["nuisance_group_id"],
            }
        )
    result = {
        "schema_version": 1,
        "artifact_type": "fluidmix_wp1_dp3_fallback_plan_index_v1",
        "status": "unrendered",
        "cases": rows,
        "render_authorized": False,
        "cross_case_visibility_aggregation": "approval_required",
        "excluded_from_model_inputs": True,
    }
    result["index_sha256"] = canonical_sha256(result)
    _write_new(output / "fallback-plan-index.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = _plan(args)
        else:
            result = register_wp1_dp3_fallback_case(
                target_ordinal=args.target_ordinal,
                manifest_root=args.manifest_root,
                repo_root=args.repo_root,
                plan_root=args.plan_root,
                case_root=args.case_root,
                host_profile_path=args.host_profile,
                offline_validation_path=args.offline_validation,
                output_path=args.case_root / "fallback-case-acceptance.json",
            )
    except (FileExistsError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"WP1 DP3 fallback {args.command} failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

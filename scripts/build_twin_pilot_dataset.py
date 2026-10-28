#!/usr/bin/env python3
"""Author, freeze, launch, or publish the frozen appearance pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix.pilot_dataset import (  # noqa: E402
    CONFIG_PATH,
    DEFAULT_ENVIRONMENT_PROFILE,
    build_experiment_graph,
    build_pilot_dataset,
    launch_pilot_attempt,
    write_attempt_manifests,
)
from fluidmix.render_process import CHILD_ENVIRONMENT_PROFILES  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    manifests = subparsers.add_parser(
        "manifests", help="author eight manifests from Task 3A inspection records"
    )
    manifests.add_argument("pilot_root")
    manifests.add_argument("--implementation-commit", required=True)
    manifests.add_argument("--controller-python", required=True)
    manifests.add_argument("--isaac-python", required=True)
    manifests.add_argument("--runtime-lock", required=True)
    manifests.add_argument("--twin", required=True)
    manifests.add_argument(
        "--environment-profile",
        choices=CHILD_ENVIRONMENT_PROFILES,
        default=DEFAULT_ENVIRONMENT_PROFILE,
    )
    manifests.add_argument("--config", default=str(CONFIG_PATH))
    manifests.add_argument("--repo-root", default=str(ROOT))
    graph = subparsers.add_parser("graph", help="hash-bind existing layers and attempt manifests")
    graph.add_argument("pilot_root")
    graph.add_argument("--implementation-commit", required=True)
    graph.add_argument("--config", default=str(CONFIG_PATH))
    graph.add_argument("--repo-root", default=str(ROOT))
    launch = subparsers.add_parser(
        "launch", help="spawn one graph-bound attempt command verbatim"
    )
    launch.add_argument("pilot_root")
    launch.add_argument("sample_id")
    launch.add_argument("--config", default=str(CONFIG_PATH))
    launch.add_argument("--repo-root", default=str(ROOT))
    dataset = subparsers.add_parser("dataset", help="publish post-review JSONLs and receipt")
    dataset.add_argument("pilot_root")
    dataset.add_argument("--visual-review", required=True)
    dataset.add_argument("--graph")
    dataset.add_argument("--config", default=str(CONFIG_PATH))
    dataset.add_argument("--repo-root", default=str(ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_status = 0
    try:
        if args.operation == "manifests":
            paths = write_attempt_manifests(
                args.pilot_root,
                config_path=args.config,
                implementation_commit=args.implementation_commit,
                controller_python=args.controller_python,
                isaac_python=args.isaac_python,
                runtime_lock=args.runtime_lock,
                twin=args.twin,
                environment_profile=args.environment_profile,
                repo_root=args.repo_root,
            )
            result = {
                "operation": "manifests",
                "paths": [str(path) for path in paths],
            }
        elif args.operation == "graph":
            path = build_experiment_graph(
                args.pilot_root,
                config_path=args.config,
                implementation_commit=args.implementation_commit,
                repo_root=args.repo_root,
            )
            result = {"operation": "graph", "path": str(path)}
        elif args.operation == "launch":
            exit_status = launch_pilot_attempt(
                args.pilot_root,
                args.sample_id,
                config_path=args.config,
                repo_root=args.repo_root,
            )
            result = {
                "operation": "launch",
                "sample_id": args.sample_id,
                "returncode": exit_status,
            }
        else:
            paths = build_pilot_dataset(
                args.pilot_root,
                config_path=args.config,
                graph_path=args.graph,
                visual_review_path=args.visual_review,
                repo_root=args.repo_root,
            )
            result = {
                "operation": "dataset",
                "paths": {key: str(value) for key, value in paths.items()},
            }
    except (FileExistsError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())

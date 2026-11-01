#!/usr/bin/env python3
"""Materialize the frozen 24-vial Brev validation-video cohort; do not render."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-plan", required=True)
    parser.add_argument("--twin", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument(
        "--isaac-python", default="/home/ubuntu/isaac/.venv/bin/python"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from twin.render.brev_video_cohort import materialize_brev_video_cohort

    graph = materialize_brev_video_cohort(
        render_plan_path=args.render_plan,
        twin_root=args.twin,
        output_dir=args.output_dir,
        runtime_lock=args.runtime_lock,
        repo_root=ROOT,
        implementation_commit=args.implementation_commit,
        isaac_python=args.isaac_python,
    )
    print(
        f"materialized {len(graph['launches'])} unrendered validation-video launches: "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

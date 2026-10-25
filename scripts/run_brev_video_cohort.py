#!/usr/bin/env python3
"""Run one materialized Brev video cohort sequentially, stopping on failure."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--controller-python", default=sys.executable)
    parser.add_argument(
        "--launcher", default=str(ROOT / "scripts/render_twin_dataset.py")
    )
    parser.add_argument(
        "--encoder", default=str(ROOT / "scripts/encode_twin_capture_video.py")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.brev_video_cohort import run_brev_video_cohort

    result = run_brev_video_cohort(
        args.cohort,
        controller_python=args.controller_python,
        launcher_script=args.launcher,
        encoder_script=args.encoder,
    )
    print(
        f"cohort {result['status']}: {result['completed_launches']}/"
        f"{result['expected_launches']} videos"
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

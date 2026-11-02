#!/usr/bin/env python3
"""Summarize a Revision 2 first train chunk and write a synthetic montage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fluidmix import fill_level_revision2_chunk as chunk  # noqa: E402


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--expected-role", choices=("train",), required=True)
    parser.add_argument("--expected-seed-start", type=int, required=True)
    parser.add_argument("--expected-seed-end", type=int, required=True)
    parser.add_argument("--expected-specs-sha256", required=True)
    parser.add_argument("--expected-renderer-sha256", required=True)
    parser.add_argument("--expected-variation-sha256", required=True)
    parser.add_argument(
        "--analysis-mode",
        choices=(chunk.DEFAULT_ANALYSIS_MODE, chunk.V4_LOOK_ANALYSIS_MODE),
        default=chunk.DEFAULT_ANALYSIS_MODE,
        help=(
            "use the existing 100-frame/F0 contract by default, or explicitly "
            "select the exact 60-frame pilot-v4 look contract"
        ),
    )
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--montage-out", type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report_out = args.report_out.expanduser().resolve()
    montage_out = args.montage_out.expanduser().resolve()
    if report_out == montage_out:
        raise ValueError("report-out and montage-out must be different paths")
    if report_out.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {report_out}")
    if montage_out.exists():
        raise FileExistsError(f"refusing to overwrite montage: {montage_out}")
    report = chunk.summarize_train_chunk(
        args.chunk_dir,
        args.expected_seed_start,
        args.expected_seed_end,
        expected_role=args.expected_role,
        montage_out=montage_out,
        expected_specs_sha256=args.expected_specs_sha256,
        expected_renderer_sha256=args.expected_renderer_sha256,
        expected_variation_sha256=args.expected_variation_sha256,
        analysis_mode=args.analysis_mode,
    )
    chunk.write_json_no_overwrite(report_out, report)
    print(
        json.dumps(
            {
                "report": str(report_out),
                "report_sha256": chunk.file_sha256(report_out),
                "montage": report["montage"],
                "rendered_liquid_hsv": report["rendered_liquid_hsv"],
                "positive_level_zero_liquid_count": len(
                    report["positive_level_zero_liquid_frames"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

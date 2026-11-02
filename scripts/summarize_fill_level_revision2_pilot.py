#!/usr/bin/env python3
"""Validate and summarize the 20-frame Step 8 Revision 2 render pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from fluidmix import fill_level_revision2_pilot as pilot  # noqa: E402


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        required=True,
        help="Copied output containing manifest.jsonl, summary.json, RGBs and masks.",
    )
    parser.add_argument(
        "--real-appearance",
        type=Path,
        required=True,
        help="Compact measurements from the 98 permitted vial-1/vial-13 frames.",
    )
    parser.add_argument(
        "--allowed-frames-root",
        type=Path,
        help="Optional indexed 98-frame calibration root for the ignored montage.",
    )
    parser.add_argument(
        "--montage-out",
        type=Path,
        help="Ignored montage destination; requires --allowed-frames-root.",
    )
    parser.add_argument(
        "--synthetic-montage-out",
        type=Path,
        help=(
            "Optional tracked montage destination containing compact synthetic "
            "crops only; never reads calibration frames."
        ),
    )
    parser.add_argument(
        "--expected-specs-sha256",
        help=(
            "Optional intended specs SHA-256; rejects a copied pilot rendered from "
            "a different specs file."
        ),
    )
    parser.add_argument("--report-out", type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    report_out = args.report_out.expanduser().resolve()
    montage_out = (
        args.montage_out.expanduser().resolve()
        if args.montage_out is not None
        else None
    )
    synthetic_montage_out = (
        args.synthetic_montage_out.expanduser().resolve()
        if args.synthetic_montage_out is not None
        else None
    )
    if montage_out is not None and montage_out == report_out:
        raise ValueError("report-out and montage-out must be different paths")
    if synthetic_montage_out is not None and synthetic_montage_out in {
        report_out,
        montage_out,
    }:
        raise ValueError(
            "synthetic-montage-out must differ from report-out and montage-out"
        )
    if report_out.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {report_out}")
    if montage_out is not None and montage_out.exists():
        raise FileExistsError(
            f"refusing to overwrite montage: {montage_out}"
        )
    if synthetic_montage_out is not None and synthetic_montage_out.exists():
        raise FileExistsError(
            "refusing to overwrite synthetic montage: "
            f"{synthetic_montage_out}"
        )
    report = pilot.summarize_pilot(
        args.pilot_dir,
        args.real_appearance,
        allowed_frames_root=args.allowed_frames_root,
        montage_out=montage_out,
        synthetic_montage_out=synthetic_montage_out,
        expected_specs_sha256=args.expected_specs_sha256,
    )
    pilot.write_json_no_overwrite(report_out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["automated_gates"]["passed"],
                "report": str(report_out),
                "report_sha256": pilot.file_sha256(report_out),
                "montage": report["montage"],
                "synthetic_montage": report["synthetic_montage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["automated_gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

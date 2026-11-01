#!/usr/bin/env python3
"""Score frozen synthetic keep/discard/abstain decisions after label separation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

from evaluation.scoring.synthetic_decision_scoring import (  # noqa: E402
    score_synthetic_stage_decision_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-evidence", required=True, type=Path)
    parser.add_argument("--measured-evidence-sha256", required=True)
    parser.add_argument("--label-provenance", required=True, type=Path)
    parser.add_argument("--label-provenance-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    score = score_synthetic_stage_decision_files(
        args.measured_evidence,
        args.label_provenance,
        args.out,
        expected_measured_evidence_sha256=args.measured_evidence_sha256,
        expected_label_provenance_sha256=args.label_provenance_sha256,
    )
    denominators = score["denominators"]
    print(
        "scored 12/12 synthetic target decisions; "
        f"covered={denominators['covered_decisions']}; "
        f"abstained={denominators['abstained_decisions']}; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

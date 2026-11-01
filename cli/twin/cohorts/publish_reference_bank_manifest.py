#!/usr/bin/env python3
"""Publish the CPU-only twin manifest consumed by the INN-67 plan builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from twin.cohorts.reference_bank_manifest import (  # noqa: E402
    publish_reference_bank_twin_artifact_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = publish_reference_bank_twin_artifact_manifest(ROOT, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

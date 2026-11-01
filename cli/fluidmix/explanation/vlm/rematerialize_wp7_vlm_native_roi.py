#!/usr/bin/env python3
"""Rematerialize accepted WP7 development full frames as native ROI crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.explanation.vlm.native_roi_rematerializer import (  # noqa: E402
    ACCEPTED_ATTEMPT09_ROOT,
    NativeRoiRematerializerError,
    rematerialize_native_roi,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ACCEPTED_ATTEMPT09_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = rematerialize_native_roi(args.source_root, args.output_root)
    except (NativeRoiRematerializerError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

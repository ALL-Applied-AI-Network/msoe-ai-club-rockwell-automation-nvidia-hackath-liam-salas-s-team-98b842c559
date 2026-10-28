#!/usr/bin/env python3
"""Write the source-bound 30-case run-varied PhysX V4 plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.physx_varied_cohort_manifest_v4 import (  # noqa: E402
    build_varied_cohort_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.out.expanduser().resolve()
    output.mkdir(mode=0o750, parents=True, exist_ok=False)
    for name, payload in build_varied_cohort_artifacts(repo_root=ROOT).items():
        (output / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

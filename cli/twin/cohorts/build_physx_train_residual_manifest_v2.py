#!/usr/bin/env python3
"""Write separated train-only leading-residue render and scoring plans."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))

from twin.cohorts.train_residual_manifest_v2 import (  # noqa: E402
    build_train_residual_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("--output-dir must be outside the Git repository")
    artifacts = build_train_residual_artifacts()
    try:
        output.mkdir(mode=0o750, parents=False, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"refusing existing output directory: {output}") from None
    except FileNotFoundError:
        raise SystemExit(f"output parent does not exist: {output.parent}") from None
    for name in (
        "render-schedule-plan.json",
        "model-input-plan.json",
        "label-provenance.json",
    ):
        temporary = output / f".{name}.tmp"
        final = output / name
        temporary.write_text(
            json.dumps(artifacts[name], indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, final)
    print(f"wrote three separated unrendered train plans: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Emit separate frozen render, model-input and label-provenance plans."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.vision_dataset_plan import CONFIG_PATH, build_vision_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit plans only; this does not render, measure, train, or run Cosmos."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory outside the repository; an existing path is refused.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("--output-dir must be outside the Git repository")
    artifacts = build_vision_artifacts(args.config, repo_root=ROOT)
    try:
        output.mkdir(mode=0o750, parents=False, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"refusing existing output directory: {output}") from None
    except FileNotFoundError:
        raise SystemExit(f"output parent does not exist: {output.parent}") from None
    for name in ("render-plan.json", "model-input-plan.json", "label-provenance-plan.json"):
        temporary = output / f".{name}.tmp"
        final = output / name
        payload = json.dumps(artifacts[name], indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, final)
    print(f"wrote three separated unrendered plan artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

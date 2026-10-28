#!/usr/bin/env python3
"""Run the frozen prepublication Lab diagnostic on mechanically valid captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluidmix.pilot_dataset import CONFIG_PATH  # noqa: E402
from fluidmix.pilot_diagnostic import evaluate_pilot_visibility  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root")
    parser.add_argument("--output")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--repo-root", default=str(ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = evaluate_pilot_visibility(
            args.pilot_root,
            output_path=args.output,
            config_path=args.config,
            repo_root=args.repo_root,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
    except (FileExistsError, ImportError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["overall_pass"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

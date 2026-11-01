#!/usr/bin/env python3
"""Verify the receipted eight-row appearance-pilot dataset fail closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twin.probes.pilot_dataset import CONFIG_PATH, check_pilot_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_root")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--repo-root", default=str(ROOT))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = check_pilot_dataset(
            args.pilot_root, config_path=args.config, repo_root=args.repo_root
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publish the immutable 72-row Varied V3 Cosmos attempt artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


def _load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--image-locator", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=ROOT / "data/cosmos_f2_varied_prompt_v1.txt",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from evaluation.scoring.cosmos_plan_artifact import (
        build_v3_attempt_plan_artifact,
    )
    from fluidmix.measurement.vision_measurement import _publish_json_no_clobber

    artifact = build_v3_attempt_plan_artifact(
        _load(args.decisions),
        args.prompt.read_text(encoding="utf-8"),
        decision_artifact_sha256=_sha(args.decisions),
        image_locator_sha256=_sha(args.image_locator),
    )
    _publish_json_no_clobber(args.out, artifact)
    print(
        f"published {artifact['denominators']['planned_attempts']} attempts to "
        f"{args.out}; sha256={_sha(args.out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Publish conservative Varied V3 triad decisions from measured evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(ROOT / "src"))


_CONFIG_FIELDS = {
    "config_id",
    "tight_peer_pair_max_delta_e76",
    "target_outlier_margin_delta_e76",
}


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def _load_json(path: Path, role: str) -> object:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"could not load {role}: {exc}") from exc


def _load_config(path: Path):
    from fluidmix.decision.triage import VariedTriadConfig

    raw = _load_json(path, "triad config")
    if not isinstance(raw, dict) or set(raw) != _CONFIG_FIELDS:
        raise ValueError("triad config fields are invalid")
    return VariedTriadConfig(
        config_id=raw["config_id"],
        tight_peer_pair_max_delta_e76=raw["tight_peer_pair_max_delta_e76"],
        target_outlier_margin_delta_e76=raw[
            "target_outlier_margin_delta_e76"
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-evidence", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from fluidmix.decision.triage import decide_varied_run_triads
    from fluidmix.measurement.vision_measurement import _publish_json_no_clobber

    measured = _load_json(args.measured_evidence, "measured evidence")
    result = decide_varied_run_triads(measured, _load_config(args.config))
    _publish_json_no_clobber(args.out, result)
    denominators = result["denominators"]
    print(
        f"assessed {denominators['vial_decisions']} vials in "
        f"{denominators['runs']} runs; "
        f"insufficient_evidence={denominators['insufficient_evidence']}; "
        f"published {args.out}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

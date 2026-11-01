#!/usr/bin/env python3
"""Build WP7 synthetic history or publish its sealed preparation preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.memory.history.history import (  # noqa: E402
    publish_preparation_preflight,
)
from fluidmix.pipeline.history_population import (  # noqa: E402
    build_development_history_population,
)


def _source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--source-database", required=True, type=Path)
    parser.add_argument("--source-database-sha256", required=True)
    parser.add_argument("--independent-review", required=True, type=Path)
    parser.add_argument("--independent-review-sha256", required=True)
    parser.add_argument("--owner-id", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare synthetic WP7 development history without one-pass access."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    population = commands.add_parser("build-population")
    _source_arguments(population)
    population.add_argument("--destination", required=True, type=Path)

    preflight = commands.add_parser("publish-preflight")
    _source_arguments(preflight)
    preflight.add_argument("--ordered-video", required=True, type=Path)
    preflight.add_argument("--ordered-video-sha256", required=True)
    preflight.add_argument("--composition-receipt", required=True, type=Path)
    preflight.add_argument("--composition-receipt-sha256", required=True)
    preflight.add_argument("--ordered-video-review", required=True, type=Path)
    preflight.add_argument("--ordered-video-review-sha256", required=True)
    preflight.add_argument("--population-receipt", required=True, type=Path)
    preflight.add_argument("--population-receipt-sha256", required=True)
    preflight.add_argument("--materialization-receipt", required=True, type=Path)
    preflight.add_argument("--materialization-receipt-sha256", required=True)
    preflight.add_argument("--development-reference-bank", required=True, type=Path)
    preflight.add_argument("--development-reference-bank-sha256", required=True)
    preflight.add_argument("--delta-e-frozen-config", required=True, type=Path)
    preflight.add_argument("--delta-e-frozen-config-sha256", required=True)
    preflight.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "source_receipt": arguments.source_receipt,
        "expected_receipt_sha256": arguments.source_receipt_sha256,
        "source_database": arguments.source_database,
        "expected_database_sha256": arguments.source_database_sha256,
        "independent_review": arguments.independent_review,
        "expected_review_sha256": arguments.independent_review_sha256,
        "owner_id": arguments.owner_id,
        "destination": arguments.destination,
    }
    if arguments.command == "build-population":
        receipt = build_development_history_population(**common)
    else:
        receipt = publish_preparation_preflight(
            **common,
            ordered_video=arguments.ordered_video,
            expected_ordered_video_sha256=arguments.ordered_video_sha256,
            composition_receipt=arguments.composition_receipt,
            expected_composition_receipt_sha256=arguments.composition_receipt_sha256,
            ordered_video_review=arguments.ordered_video_review,
            expected_ordered_video_review_sha256=arguments.ordered_video_review_sha256,
            population_receipt=arguments.population_receipt,
            expected_population_receipt_sha256=arguments.population_receipt_sha256,
            materialization_receipt=arguments.materialization_receipt,
            expected_materialization_receipt_sha256=arguments.materialization_receipt_sha256,
            development_reference_bank=arguments.development_reference_bank,
            expected_development_reference_bank_sha256=arguments.development_reference_bank_sha256,
            delta_e_frozen_config=arguments.delta_e_frozen_config,
            expected_delta_e_frozen_config_sha256=arguments.delta_e_frozen_config_sha256,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

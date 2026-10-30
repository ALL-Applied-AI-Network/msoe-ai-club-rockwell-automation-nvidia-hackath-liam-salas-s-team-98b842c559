#!/usr/bin/env python3
"""Thin command line entry point for the production carryover scaffold."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fluidmix.carryover_pipeline import (  # noqa: E402
    freeze_validation,
    load_preflight_receipt,
    run_real_check,
    run_test,
)
from fluidmix.wp7_rgb_runtime import (  # noqa: E402
    FrozenCNNPredictedMaskRuntime,
    load_frozen_cnn,
)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--recipe-sequence", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument(
        "--selected-mode",
        required=True,
        choices=("evaluation_only", "history_backed"),
    )
    parser.add_argument("--cnn-checkpoint", required=True, type=Path)
    parser.add_argument("--cnn-checkpoint-sha256", required=True)
    parser.add_argument("--cnn-replay", required=True, type=Path)
    parser.add_argument("--cnn-replay-sha256", required=True)
    parser.add_argument("--cnn-review", required=True, type=Path)
    parser.add_argument("--cnn-review-sha256", required=True)
    parser.add_argument("--capture-receipt", required=True, type=Path)
    parser.add_argument("--capture-receipt-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze validation or seal a one-pass carryover production receipt."
    )
    phases = parser.add_subparsers(dest="phase", required=True)
    freeze_parser = phases.add_parser("freeze-validation")
    freeze_parser.add_argument("--validation-results", required=True, type=Path)
    freeze_parser.add_argument("--out", required=True, type=Path)
    _run_arguments(phases.add_parser("run-test"))
    _run_arguments(phases.add_parser("run-real-check"))
    return parser


def _reject_preflight_path_alias(arguments: argparse.Namespace) -> None:
    preflight_path = arguments.preflight.resolve(strict=False)
    phase_paths = {
        "video": arguments.video,
        "recipe sequence": arguments.recipe_sequence,
        "freeze": arguments.freeze,
        "store": arguments.store,
        "output": arguments.out,
        "claim": arguments.claim,
        "journal": Path(f"{arguments.out}.journal.jsonl"),
        "CNN checkpoint": arguments.cnn_checkpoint,
        "CNN replay": arguments.cnn_replay,
        "CNN review": arguments.cnn_review,
        "capture receipt": arguments.capture_receipt,
    }
    for label, path in phase_paths.items():
        if preflight_path == path.resolve(strict=False):
            raise ValueError(f"preflight path must not alias the {label} path")
    all_paths = list(phase_paths.items()) + [("preflight", arguments.preflight)]
    resolved = [path.resolve(strict=False) for _, path in all_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("production, runtime and operational paths must be distinct")


def _runtime_loader(arguments: argparse.Namespace):
    """Defer every runtime-file read until the controller has acquired the claim."""

    def load() -> FrozenCNNPredictedMaskRuntime:
        loaded = load_frozen_cnn(
            checkpoint_path=arguments.cnn_checkpoint.absolute(),
            replay_path=arguments.cnn_replay.absolute(),
            review_path=arguments.cnn_review.absolute(),
            expected_checkpoint_sha256=arguments.cnn_checkpoint_sha256,
            expected_replay_sha256=arguments.cnn_replay_sha256,
            expected_review_sha256=arguments.cnn_review_sha256,
        )
        receipt_path = arguments.capture_receipt.absolute()
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("capture receipt must be a regular non-symlink file")
        raw = receipt_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != arguments.capture_receipt_sha256:
            raise ValueError("capture receipt SHA-256 mismatch")
        try:
            receipt = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("capture receipt must be UTF-8 JSON") from error
        return FrozenCNNPredictedMaskRuntime(loaded, receipt)

    return load


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.phase == "freeze-validation":
        receipt = freeze_validation(arguments.validation_results, arguments.out)
    else:
        function = run_test if arguments.phase == "run-test" else run_real_check
        _reject_preflight_path_alias(arguments)
        preflight_receipt = load_preflight_receipt(arguments.preflight)
        receipt = function(
            video=arguments.video,
            recipe_sequence=arguments.recipe_sequence,
            freeze=arguments.freeze,
            expected_freeze_sha256=arguments.freeze_sha256,
            store=arguments.store,
            out=arguments.out,
            claim=arguments.claim,
            owner_id=arguments.owner_id,
            preflight_receipt=preflight_receipt,
            expected_preflight_sha256=arguments.preflight_sha256,
            selected_mode=arguments.selected_mode,
            mask_runtime_loader=_runtime_loader(arguments),
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

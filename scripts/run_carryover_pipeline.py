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
from fluidmix.wp7_phase_capture import (  # noqa: E402
    decode_video_rgb_frames,
    publish_phase_capture,
)
from fluidmix.wp7_development_reference import DevelopmentReferenceStore  # noqa: E402
from fluidmix.wp7_cohort_preparation import (  # noqa: E402
    require_video5_runtime_capacity,
)
from fluidmix.wp7_fill_step_production import (  # noqa: E402
    load_pixel_to_metre_calibration,
    load_topology_freeze,
    load_tracker_freeze,
)
from fluidmix.wp7_tracker_runtime import (  # noqa: E402
    FrozenFillStepTrackerRuntime,
    TrackerLineage,
)


def _run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--recipe-sequence", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--evidence-out-store", required=True, type=Path)
    parser.add_argument("--development-reference-store", type=Path)
    parser.add_argument("--development-reference-store-sha256")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument(
        "--selected-mode",
        required=True,
        choices=(
            "evaluation_only",
            "history_backed",
            "development_reference_backed",
            "same_camera_cohort_goldens",
        ),
    )
    parser.add_argument("--cnn-checkpoint", required=True, type=Path)
    parser.add_argument("--cnn-checkpoint-sha256", required=True)
    parser.add_argument("--cnn-replay", required=True, type=Path)
    parser.add_argument("--cnn-replay-sha256", required=True)
    parser.add_argument("--cnn-review", required=True, type=Path)
    parser.add_argument("--cnn-review-sha256", required=True)
    parser.add_argument("--capture-plan", required=True, type=Path)
    parser.add_argument("--capture-plan-sha256", required=True)
    parser.add_argument("--capture-receipt-out", required=True, type=Path)
    parser.add_argument("--capture-provenance-out", required=True, type=Path)
    parser.add_argument("--topology-freeze", type=Path)
    parser.add_argument("--topology-freeze-sha256")
    parser.add_argument("--tracker-train-freeze", type=Path)
    parser.add_argument("--tracker-train-freeze-sha256")
    parser.add_argument("--tracker-materialization", type=Path)
    parser.add_argument("--tracker-materialization-sha256")
    parser.add_argument("--pixel-calibration", type=Path)
    parser.add_argument("--pixel-calibration-sha256")
    parser.add_argument("--tracker-sidecar-out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--initial-previous-final-tank-id",
        choices=("unknown", "t100", "t200", "t400"),
    )


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
        "evidence output store": arguments.evidence_out_store,
        "output": arguments.out,
        "claim": arguments.claim,
        "journal": Path(f"{arguments.out}.journal.jsonl"),
        "CNN checkpoint": arguments.cnn_checkpoint,
        "CNN replay": arguments.cnn_replay,
        "CNN review": arguments.cnn_review,
        "capture plan": arguments.capture_plan,
        "capture receipt output": arguments.capture_receipt_out,
        "capture provenance output": arguments.capture_provenance_out,
    }
    if arguments.development_reference_store is not None:
        phase_paths["development reference store"] = arguments.development_reference_store
    for label, path in phase_paths.items():
        if preflight_path == path.resolve(strict=False):
            raise ValueError(f"preflight path must not alias the {label} path")
    all_paths = list(phase_paths.items()) + [("preflight", arguments.preflight)]
    resolved = [path.resolve(strict=False) for _, path in all_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("production, runtime and operational paths must be distinct")


def _runtime_loader(arguments: argparse.Namespace):
    """Defer every runtime-file read until the controller has acquired the claim."""

    def load(
        video_bytes: bytes,
        recipe_sequence: dict[str, object],
        phase: str,
        video_sha256: str,
    ) -> FrozenCNNPredictedMaskRuntime:
        tracker_mode = (
            phase == "run-real-check"
            and arguments.selected_mode == "same_camera_cohort_goldens"
        )
        # Every hash-bound model/tracker input is opened and validated before
        # the first camera frame is decoded.
        loaded = load_frozen_cnn(
            checkpoint_path=arguments.cnn_checkpoint.absolute(),
            replay_path=arguments.cnn_replay.absolute(),
            review_path=arguments.cnn_review.absolute(),
            expected_checkpoint_sha256=arguments.cnn_checkpoint_sha256,
            expected_replay_sha256=arguments.cnn_replay_sha256,
            expected_review_sha256=arguments.cnn_review_sha256,
        )
        topology = calibration = tracker_freeze = None
        recipe_semantic_sha256 = hashlib.sha256(json.dumps(
            recipe_sequence, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode()).hexdigest()
        if tracker_mode:
            topology = load_topology_freeze(
                arguments.topology_freeze.absolute(),
                expected_sha256=arguments.topology_freeze_sha256,
            )
            calibration = load_pixel_to_metre_calibration(
                arguments.pixel_calibration.absolute(),
                expected_sha256=arguments.pixel_calibration_sha256,
            )
            tracker_freeze = load_tracker_freeze(
                arguments.tracker_materialization.absolute(),
                expected_materialization_sha256=arguments.tracker_materialization_sha256,
                train_freeze_path=arguments.tracker_train_freeze.absolute(),
                expected_train_freeze_sha256=arguments.tracker_train_freeze_sha256,
                topology=topology, calibration=calibration,
                target_recipe_artifact_sha256=recipe_semantic_sha256,
            )
        frames = decode_video_rgb_frames(video_bytes)
        publication = publish_phase_capture(
            plan_path=arguments.capture_plan.absolute(),
            expected_plan_sha256=arguments.capture_plan_sha256,
            expected_phase=phase,
            expected_capture_origin=(
                "real_camera" if phase == "run-real-check" else "synthetic_camera"
            ),
            video_bytes=video_bytes,
            recipe_sequence=recipe_sequence,
            decoded_rgb_frames=frames,
            capture_receipt_out=arguments.capture_receipt_out.absolute(),
            provenance_out=arguments.capture_provenance_out.absolute(),
        )
        if publication.capture_receipt["video_sha256"] != video_sha256:
            raise ValueError("published capture video SHA-256 differs from claimed video")
        if tracker_mode:
            assert topology is not None and calibration is not None and tracker_freeze is not None
            return FrozenFillStepTrackerRuntime(
                loaded_cnn=loaded, capture_receipt=publication.capture_receipt,
                freeze=tracker_freeze, topology=topology,
                lineage=TrackerLineage(
                    topology_file_sha256=arguments.topology_freeze_sha256,
                    topology_receipt_sha256=topology.receipt_sha256,
                    train_freeze_file_sha256=arguments.tracker_train_freeze_sha256,
                    materialization_file_sha256=arguments.tracker_materialization_sha256,
                    calibration_file_sha256=arguments.pixel_calibration_sha256,
                    checkpoint_sha256=arguments.cnn_checkpoint_sha256,
                    replay_sha256=arguments.cnn_replay_sha256,
                    review_sha256=arguments.cnn_review_sha256,
                ),
                sidecar_out=arguments.tracker_sidecar_out,
                device=arguments.device,
                target_recipe_raw_sha256=hashlib.sha256(
                    arguments.recipe_sequence.read_bytes()
                ).hexdigest(),
                target_recipe_semantic_sha256=recipe_semantic_sha256,
                initial_previous_final_tank_id=(
                    None if arguments.initial_previous_final_tank_id == "unknown"
                    else arguments.initial_previous_final_tank_id
                ),
                frame_decoder=lambda _raw: publication.decoded_rgb_frames,
            )
        return FrozenCNNPredictedMaskRuntime(
            loaded,
            publication.capture_receipt,
            _frame_decoder=lambda _raw: publication.decoded_rgb_frames,
        )

    return load


def _development_reference_loader(arguments: argparse.Namespace):
    """Defer donor-store open, SHA, schema and integrity checks until the controller asks."""

    if arguments.development_reference_store is None:
        return None

    def load() -> DevelopmentReferenceStore:
        return DevelopmentReferenceStore(
            arguments.development_reference_store.absolute(),
            expected_snapshot_sha256=arguments.development_reference_store_sha256,
        )

    return load


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.phase == "freeze-validation":
        receipt = freeze_validation(arguments.validation_results, arguments.out)
    else:
        function = run_test if arguments.phase == "run-test" else run_real_check
        has_reference_path = arguments.development_reference_store is not None
        has_reference_sha = arguments.development_reference_store_sha256 is not None
        if arguments.selected_mode == "development_reference_backed":
            if not has_reference_path or not has_reference_sha:
                raise ValueError(
                    "development_reference_backed mode requires both development "
                    "reference store arguments"
                )
        elif has_reference_path or has_reference_sha:
            raise ValueError(
                "development reference store arguments require "
                "development_reference_backed mode"
            )
        if (
            arguments.phase == "run-real-check"
            and arguments.selected_mode == "same_camera_cohort_goldens"
        ):
            tracker_fields = (
                "topology_freeze", "topology_freeze_sha256",
                "tracker_train_freeze", "tracker_train_freeze_sha256",
                "tracker_materialization", "tracker_materialization_sha256",
                "pixel_calibration", "pixel_calibration_sha256",
                "tracker_sidecar_out", "device", "initial_previous_final_tank_id",
            )
            missing = [field for field in tracker_fields if getattr(arguments, field) is None]
            if missing:
                raise ValueError(
                    "same_camera_cohort_goldens requires tracker input: " + missing[0]
                )
            require_video5_runtime_capacity(output_path=arguments.out)
        _reject_preflight_path_alias(arguments)
        preflight_receipt = load_preflight_receipt(arguments.preflight)
        receipt = function(
            video=arguments.video,
            recipe_sequence=arguments.recipe_sequence,
            freeze=arguments.freeze,
            expected_freeze_sha256=arguments.freeze_sha256,
            evidence_out_store=arguments.evidence_out_store,
            out=arguments.out,
            claim=arguments.claim,
            owner_id=arguments.owner_id,
            preflight_receipt=preflight_receipt,
            expected_preflight_sha256=arguments.preflight_sha256,
            selected_mode=arguments.selected_mode,
            capture_plan=arguments.capture_plan,
            capture_plan_sha256=arguments.capture_plan_sha256,
            capture_receipt_out=arguments.capture_receipt_out,
            capture_provenance_out=arguments.capture_provenance_out,
            capture_runtime_loader=_runtime_loader(arguments),
            development_reference_store=arguments.development_reference_store,
            development_reference_store_sha256=arguments.development_reference_store_sha256,
            development_reference_store_loader=_development_reference_loader(arguments),
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Metadata-only preparation for the accepted WP7 cohort-goldens route."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

from .carryover_pipeline import (
    RECIPE_SEQUENCE_SCHEMA_SHA256,
    freeze_validation,
    pipeline_source_sha256,
    preflight_explanation_route,
)
from .predicted_mask_materializer import (
    CNN_CHECKPOINT_SHA256,
    CNN_REPLAY_SHA256,
    CNN_REVIEW_SHA256,
)
from .wp7_rgb_runtime import (
    FrozenCNNPredictedMaskRuntime,
    RGBRuntimeError,
    compose_fill_windows,
    load_frozen_cnn,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ROUTE_DECISION = {
    "schema_version": "wp7-same-camera-cohort-goldens-route-decision-v1",
    "artifact_type": "wp7_same_camera_cohort_goldens_route_decision",
    "issue_id": "INN-67",
    "scope": "wp1_and_video5_same_camera_cohort_goldens",
    "approved_mode": "same_camera_cohort_goldens",
    "decision_status": "approved",
    "approver_name": "Liam",
    "approver_handle": "@salasl",
    "reference_rule": (
        "closed_cohort_position_1_compares_to_golden_positions_2_and_3_"
        "positions_2_and_3_reference_control"
    ),
    "distance_metric": "CIE76",
    "keep_threshold_delta_e76": 1.0,
    "discard_threshold_delta_e76": 2.0,
    "aggregation_rule": (
        "keep_every_stage_both_le_1_discard_any_stage_both_ge_2_else_insufficient"
    ),
    "temporal_scope": "post_complete_same_run_three_vial_cohort_only",
    "evidence_boundary": (
        "runtime_rgb_cnn_measurements_and_programmed_recipe_metadata_only"
    ),
}


@dataclass(frozen=True)
class CohortPreparation:
    publication_receipt: Mapping[str, object]
    output_paths: tuple[Path, ...]


@dataclass(frozen=True)
class FreezePreparation:
    validation_results_path: Path
    freeze_path: Path
    preparation_receipt_path: Path


def _bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _sha(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{role} must be a lowercase SHA-256")
    return value


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _bound_json(path_value: Path, expected: object, role: str) -> tuple[Path, bytes, dict[str, object]]:
    path = Path(path_value).expanduser().absolute()
    digest = _sha(expected, f"{role} SHA-256")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{role} must be a regular non-symlink file") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{role} must be a regular non-symlink file")
        raw = b""
        while chunk := os.read(descriptor, 1024 * 1024):
            raw += chunk
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{role} raw SHA-256 mismatch")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return path, raw, value


def _bound_bytes(path_value: Path, expected: object, role: str) -> tuple[Path, bytes]:
    path = Path(path_value).expanduser().absolute()
    digest = _sha(expected, f"{role} SHA-256")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{role} must be a regular non-symlink file") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"{role} raw SHA-256 mismatch")
    return path, raw


def _case_ids(sequence: Mapping[str, object]) -> tuple[list[str], list[str]]:
    if (
        sequence.get("schema_version") != "carryover-recipe-sequence-v1"
        or sequence.get("split_role") != "test"
        or not isinstance(sequence.get("runs"), list)
        or len(sequence["runs"]) != 2
    ):
        raise ValueError("recipe sequence identity is invalid")
    runs: list[str] = []
    cases: list[str] = []
    for run in sequence["runs"]:
        if not isinstance(run, Mapping) or not isinstance(run.get("vials"), list):
            raise ValueError("recipe sequence run identity is invalid")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in runs:
            raise ValueError("recipe sequence run order is invalid")
        runs.append(run_id)
        for vial in run["vials"]:
            case = vial.get("case_id") if isinstance(vial, Mapping) else None
            if not isinstance(case, str) or not case or case in cases:
                raise ValueError("recipe sequence case identity/order is invalid")
            cases.append(case)
    if not cases:
        raise ValueError("recipe sequence has no cases")
    return runs, cases


def _transition(row: object, index: int, frame_count: int, role: str) -> dict[str, int | None]:
    if not isinstance(row, Mapping):
        raise ValueError(f"{role} timing is invalid")
    segment = row.get("segment_index")
    start = row.get("start_frame_index")
    stop = row.get("stop_frame_index")
    count = row.get("frame_count")
    frame_hashes = row.get("decoded_bgr_sha256")
    if (
        type(segment) is not int or segment != index
        or type(start) is not int or type(stop) is not int or type(count) is not int
        or start < 0 or stop <= start or stop > frame_count or count != stop - start
        or not isinstance(frame_hashes, list) or len(frame_hashes) != count
        or any(not isinstance(item, str) or _SHA256.fullmatch(item) is None
               for item in frame_hashes)
    ):
        raise ValueError(f"{role} timing is invalid")
    return {
        "segment_index": segment, "start_frame_index": start,
        "stop_frame_index": stop, "frame_count": count,
        "decoded_bgr_sha256": None,
    }


def _prepare_geometry(
    run_values: Sequence[Mapping[str, object]], case_ids: list[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]], str, int]:
    transitions: list[dict[str, object]] = []
    fills: list[dict[str, object]] = []
    offset = 0
    for run_index, run in enumerate(run_values):
        frame_count = run.get("output_frame_count")
        run_transitions = run.get("encoded_transition_evidence")
        run_fills = run.get("fill_clips")
        if (
            type(frame_count) is not int or frame_count <= 0
            or not isinstance(run_transitions, list) or len(run_transitions) < 2
            or not isinstance(run_fills, list) or not run_fills
        ):
            raise ValueError("run timing identity is invalid")
        normalized = [
            _transition(row, index, frame_count, f"run {run_index + 1}")
            for index, row in enumerate(run_transitions)
        ]
        normalized_fills: list[dict[str, object]] = []
        for fill in run_fills:
            if (
                not isinstance(fill, Mapping)
                or not isinstance(fill.get("case_id"), str)
                or type(fill.get("frame_count")) is not int
                or int(fill["frame_count"]) <= 0
            ):
                raise ValueError("run fill timing identity is invalid")
            normalized_fills.append({
                "case_id": fill["case_id"], "frame_count": fill["frame_count"]
            })
        fills.extend(normalized_fills)
        if run_index:
            previous = transitions.pop()
            entry = normalized.pop(0)
            if previous["stop_frame_index"] != offset or entry["start_frame_index"] != 0:
                raise ValueError("run boundary timing is invalid")
            merged_start = int(previous["start_frame_index"])
            merged_stop = offset + int(entry["stop_frame_index"])
            merged_count = int(previous["frame_count"]) + int(entry["frame_count"])
            if merged_stop - merged_start != merged_count:
                raise ValueError("run boundary timing is invalid")
            transitions.append({
                "segment_index": len(transitions),
                "start_frame_index": merged_start,
                "stop_frame_index": merged_stop,
                "frame_count": merged_count,
                "decoded_bgr_sha256": None,
            })
        for row in normalized:
            transitions.append({
                **row,
                "segment_index": len(transitions),
                "start_frame_index": int(row["start_frame_index"]) + offset,
                "stop_frame_index": int(row["stop_frame_index"]) + offset,
            })
        offset += frame_count
    if [fill["case_id"] for fill in fills] != case_ids:
        raise ValueError("run fill case order differs from recipe sequence")
    timing_transitions = [
        {
            **row,
            "decoded_bgr_sha256": ["0" * 64] * int(row["frame_count"]),
        }
        for row in transitions
    ]
    try:
        windows = compose_fill_windows(timing_transitions, fills)
    except RGBRuntimeError as error:
        raise ValueError("combined run timing identity is invalid") from error
    timing = {
        "fill_clips": fills,
        "transition_ranges": [
            {key: row[key] for key in (
                "segment_index", "start_frame_index", "stop_frame_index", "frame_count"
            )}
            for row in transitions
        ],
        "fill_windows": [
            {
                "fill_index": window.fill_index, "case_id": window.case_id,
                "start_frame_index": window.start_frame_index,
                "stop_frame_index": window.stop_frame_index,
                "frame_count": window.frame_count,
            }
            for window in windows
        ],
    }
    return transitions, fills, _hash(timing), offset


def _write(path: Path, value: object) -> str:
    payload = _bytes(value) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def prepare_run_test_inputs(
    *,
    recipe_sequence: Path,
    expected_recipe_sequence_sha256: str,
    composition_receipt: Path,
    expected_composition_receipt_sha256: str,
    final_review: Path,
    expected_final_review_sha256: str,
    run_receipts: Sequence[Path],
    expected_run_receipt_sha256s: Sequence[str],
    destination: Path,
) -> CohortPreparation:
    """Publish preparation metadata without opening any declared video path."""

    if len(run_receipts) != 2 or len(expected_run_receipt_sha256s) != 2:
        raise ValueError("exactly two ordered run receipts are required")
    sequence_path, _, sequence = _bound_json(
        recipe_sequence, expected_recipe_sequence_sha256, "recipe sequence"
    )
    composition_path, _, composition = _bound_json(
        composition_receipt, expected_composition_receipt_sha256,
        "composition receipt",
    )
    review_path, _, review = _bound_json(
        final_review, expected_final_review_sha256, "final review"
    )
    loaded_runs = [
        _bound_json(path, digest, f"run receipt {index + 1}")
        for index, (path, digest) in enumerate(
            zip(run_receipts, expected_run_receipt_sha256s, strict=True)
        )
    ]
    run_ids, case_ids = _case_ids(sequence)
    if len(case_ids) != 18 or any(
        len(run["vials"]) != 9 for run in sequence["runs"]
    ):
        raise ValueError("accepted WP1 geometry requires two nine-vial runs")
    ordered_runs = composition.get("ordered_runs")
    output = composition.get("output")
    verification = composition.get("verification")
    if (
        composition.get("artifact_type") != "wp7_ordered_test_video_v1"
        or composition.get("status") != "pass"
        or not isinstance(ordered_runs, list) or len(ordered_runs) != 2
        or not isinstance(output, Mapping)
        or not isinstance(verification, Mapping)
        or verification.get("case_count") != len(case_ids)
        or verification.get("run_count") != 2
        or verification.get("source_order") != run_ids
    ):
        raise ValueError("composition identity/order is invalid")
    video_sha = _sha(output.get("sha256"), "composition video")
    frame_count = output.get("frame_count")
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("composition frame count is invalid")
    run_values: list[Mapping[str, object]] = []
    manifest_bindings: list[dict[str, object]] = []
    for index, ((run_path, _, run), binding) in enumerate(
        zip(loaded_runs, ordered_runs, strict=True)
    ):
        if not isinstance(binding, Mapping):
            raise ValueError("composition run binding is invalid")
        expected = {
            "path": str(run_path),
            "sha256": expected_run_receipt_sha256s[index],
            "artifact_type": "rendered_transport_run_sequence_v4",
            "run_id": run_ids[index],
            "case_ids": [vial["case_id"] for vial in sequence["runs"][index]["vials"]],
            "output_video_sha256": binding.get("video_sha256"),
            "output_frame_count": binding.get("frame_count"),
        }
        if (
            binding.get("run_receipt_path") != str(run_path)
            or binding.get("run_receipt_sha256") != expected_run_receipt_sha256s[index]
            or run.get("artifact_type") != expected["artifact_type"]
            or run.get("status") != "pass"
            or run.get("run_id") != expected["run_id"]
            or run.get("case_ids") != expected["case_ids"]
            or run.get("output_video_sha256") != expected["output_video_sha256"]
            or run.get("output_frame_count") != expected["output_frame_count"]
        ):
            raise ValueError("composition/run identity, order, or case binding differs")
        manifest_bindings.append(expected)
        run_values.append(run)
    if (
        [case for binding in ordered_runs for case in binding.get("case_ids", [])]
        != case_ids
    ):
        raise ValueError("composition case order differs")
    transitions, fills, timing_sha, combined_count = _prepare_geometry(
        run_values, case_ids
    )
    if combined_count != frame_count:
        raise ValueError("composition and run frame counts differ")
    if (
        review.get("artifact_type")
        != "fluidmix_wp7_ordered_test_video_final_output_review_v1"
        or review.get("status") != "pass" or review.get("verdict") != "pass"
        or not isinstance(review.get("identity"), Mapping)
        or review["identity"].get("output_mp4_sha256") != video_sha
        or review["identity"].get("output_receipt_sha256")
        != expected_composition_receipt_sha256
        or not isinstance(review.get("case_contract"), Mapping)
        or review["case_contract"].get("count") != len(case_ids)
        or review["case_contract"].get("unique") is not True
        or not isinstance(review.get("output_media"), Mapping)
        or review["output_media"].get("sha256") != video_sha
        or review["output_media"].get("frame_count") != frame_count
    ):
        raise ValueError("final review identity is invalid")

    route = dict(_ROUTE_DECISION)
    preflight = preflight_explanation_route(
        requested_mode="same_camera_cohort_goldens", expected_test_vials=18,
        expected_real_check_vials=30, route_decision=route,
        expected_route_decision_sha256=_hash(route),
    )
    destination_path = Path(destination).expanduser().absolute()
    destination_path.mkdir(parents=True, exist_ok=False)
    try:
        manifest_path = destination_path / "run-receipts.json"
        manifest = {
            "schema_version": 1,
            "artifact_type": "wp7_ordered_test_run_receipt_v1",
            "status": "pass", "run_receipts": manifest_bindings,
        }
        manifest_sha = _write(manifest_path, manifest)
        recipe_canonical_sha = _hash(sequence)
        sources_spec = (
            ("ordered_wp1_composition_receipt", composition_path,
             expected_composition_receipt_sha256, "wp7_ordered_test_video_v1",
             "source-01-composition.json"),
            ("ordered_wp1_review_receipt", review_path,
             expected_final_review_sha256,
             "fluidmix_wp7_ordered_test_video_final_output_review_v1",
             "source-02-review.json"),
            ("ordered_wp1_run_receipt", manifest_path, manifest_sha,
             "wp7_ordered_test_run_receipt_v1", "source-03-runs.json"),
        )
        source_rows: list[dict[str, str]] = []
        output_hashes: dict[str, str] = {manifest_path.name: manifest_sha}
        for role, upstream_path, upstream_sha, artifact_type, filename in sources_spec:
            adapter = {
                "schema_version": "wp7-capture-source-adapter-v1",
                "artifact_type": "wp7_capture_source_adapter",
                "phase": "run-test", "role": role, "video_sha256": video_sha,
                "recipe_sequence_sha256": recipe_canonical_sha,
                "ordered_case_ids": case_ids,
                "capture_timing_sha256": timing_sha,
                "upstream_artifact_type": artifact_type,
                "upstream_path": str(upstream_path),
                "upstream_sha256": upstream_sha, "status": "pass",
            }
            adapter_path = destination_path / filename
            digest = _write(adapter_path, adapter)
            output_hashes[filename] = digest
            source_rows.append({"role": role, "path": str(adapter_path), "sha256": digest})
        plan_path = destination_path / "capture-plan.json"
        plan = {
            "schema_version": "wp7-phase-capture-plan-v1",
            "artifact_type": "wp7_phase_capture_plan", "phase": "run-test",
            "capture_origin": "synthetic_camera", "video_sha256": video_sha,
            "fill_clips": fills, "encoded_transition_evidence": transitions,
            "source_provenance": source_rows,
        }
        output_hashes[plan_path.name] = _write(plan_path, plan)
        route_path = destination_path / "route-decision.json"
        output_hashes[route_path.name] = _write(route_path, route)
        preflight_path = destination_path / "execution-preflight.json"
        output_hashes[preflight_path.name] = _write(preflight_path, preflight)
        receipt_path = destination_path / "publication-receipt.json"
        receipt = {
            "schema_version": "wp7-cohort-preparation-publication-v1",
            "artifact_type": "wp7_cohort_preparation_publication",
            "status": "pass", "access_policy": "metadata_only_no_media_access",
            "inputs": {
                "recipe_sequence": {"path": str(sequence_path),
                                    "raw_sha256": expected_recipe_sequence_sha256,
                                    "canonical_sha256": recipe_canonical_sha},
                "composition_receipt": {"path": str(composition_path),
                                        "raw_sha256": expected_composition_receipt_sha256},
                "final_review": {"path": str(review_path),
                                 "raw_sha256": expected_final_review_sha256},
                "run_receipts": [
                    {"path": str(item[0]), "raw_sha256": digest}
                    for item, digest in zip(loaded_runs, expected_run_receipt_sha256s, strict=True)
                ],
                "declared_video": {"sha256": video_sha, "frame_count": frame_count,
                                   "path_accessed": False},
            },
            "canonical_bindings": {
                "route_decision_sha256": _hash(route),
                "execution_preflight_sha256": preflight["preflight_sha256"],
                "capture_timing_sha256": timing_sha,
            },
            "outputs": dict(sorted(output_hashes.items())),
        }
        receipt["publication_sha256"] = _hash(receipt)
        output_hashes[receipt_path.name] = _write(receipt_path, receipt)
        paths = tuple(destination_path / name for name in (
            "run-receipts.json", "source-01-composition.json", "source-02-review.json",
            "source-03-runs.json", "capture-plan.json", "route-decision.json",
            "execution-preflight.json", "publication-receipt.json",
        ))
        return CohortPreparation(receipt, paths)
    except BaseException:
        shutil.rmtree(destination_path)
        raise


def _source_hash(relative: str) -> str:
    path = Path(__file__).resolve().parents[2] / relative
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_validation_freeze(
    *,
    checkpoint: Path,
    expected_checkpoint_sha256: str,
    replay: Path,
    expected_replay_sha256: str,
    review: Path,
    expected_review_sha256: str,
    destination: Path,
) -> FreezePreparation:
    """Derive the accepted validation freeze without opening any camera media."""

    if expected_checkpoint_sha256 != CNN_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 differs from accepted lineage")
    if expected_replay_sha256 != CNN_REPLAY_SHA256:
        raise ValueError("replay SHA-256 differs from accepted lineage")
    if expected_review_sha256 != CNN_REVIEW_SHA256:
        raise ValueError("review SHA-256 differs from accepted lineage")
    checkpoint_path, _ = _bound_bytes(
        checkpoint, expected_checkpoint_sha256, "CNN checkpoint"
    )
    replay_path, _, replay_value = _bound_json(
        replay, expected_replay_sha256, "CNN replay"
    )
    review_path, review_raw = _bound_bytes(review, expected_review_sha256, "CNN review")
    try:
        review_text = review_raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("CNN review must be UTF-8 metadata") from error
    if (
        replay_value.get("artifact_type") != "wp5_cnn_deterministic_bounds_replay"
        or replay_value.get("status") != "swap_review_ready"
        or replay_value.get("execution_device") != "cpu"
        or replay_value.get("threshold_frozen") != 0.5
        or replay_value.get("checkpoint_frozen") is not True
        or f"`{expected_replay_sha256}`" not in review_text
        or "PASS" not in review_text or "CPU" not in review_text.upper()
    ):
        raise ValueError("accepted replay/review status or CPU identity is invalid")
    source_bindings = replay_value.get("source_bindings")
    checkpoint_binding = (
        source_bindings.get("checkpoint") if isinstance(source_bindings, Mapping) else None
    )
    if (
        not isinstance(checkpoint_binding, Mapping)
        or checkpoint_binding.get("sha256") != expected_checkpoint_sha256
    ):
        raise ValueError("replay runtime lineage differs from the accepted checkpoint")
    rows = replay_value.get("replayed_rows")
    determinism = replay_value.get("determinism_gate")
    scoring = replay_value.get("scoring")
    scoring_denominators = (
        scoring.get("denominators") if isinstance(scoring, Mapping) else None
    )
    if (
        not isinstance(rows, list) or len(rows) != 21
        or not isinstance(determinism, Mapping)
        or determinism.get("matched_rows") != 21
        or determinism.get("mismatch_count") != 0
        or determinism.get("status") != "pass"
        or not isinstance(scoring_denominators, Mapping)
        or scoring_denominators.get("all_positive_rows") != 21
    ):
        raise ValueError("accepted replay validation denominator/status is invalid")
    run_ids: set[str] = set()
    vial_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("accepted replay row identity is invalid")
        run_id, case_stage = row.get("run_id"), row.get("case_id")
        if not isinstance(run_id, str) or not isinstance(case_stage, str) or ":" not in case_stage:
            raise ValueError("accepted replay row identity is invalid")
        run_ids.add(run_id)
        vial_ids.add(case_stage.split(":", 1)[0])
    if len(run_ids) != 3 or len(vial_ids) != 9:
        raise ValueError("accepted replay run/vial denominator is invalid")

    loaded = load_frozen_cnn(
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        replay_path=replay_path,
        review_path=review_path,
        expected_replay_sha256=expected_replay_sha256,
        expected_review_sha256=expected_review_sha256,
    )
    shell = {
        "schema_version": "wp7-camera-capture-v1",
        "artifact_type": "wp7_camera_capture",
        "capture_origin": "synthetic_camera",
        "video_sha256": _hash({
            "purpose": "metadata-only runtime identity shell",
            "checkpoint_sha256": expected_checkpoint_sha256,
            "replay_sha256": expected_replay_sha256,
        }),
        "encoded_transition_evidence": [], "fill_clips": [],
    }
    shell["receipt_sha256"] = _hash(shell)
    runtime = FrozenCNNPredictedMaskRuntime(loaded, shell)

    decision_config = {
        "discard_threshold_delta_e76": 2.0,
        "keep_threshold_delta_e76": 1.0,
    }
    guard_policy = {
        "method": "deterministic_guarded_template",
        "model_id": "none",
        "unsupported_claim_behavior": "insufficient_evidence",
    }
    measurement_sha = _source_hash("src/fluidmix/vision_measurement.py")
    explanation_source_sha = _source_hash("src/fluidmix/carryover_pipeline.py")
    store_sha = _source_hash("src/fluidmix/evidence_store.py")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("pipeline source commit is unavailable") from error
    validation = {
        "schema_version": "carryover-validation-results-v1",
        "artifact_type": "carryover_validation_results",
        "split_role": "validation", "pipeline_source_commit": commit,
        "pipeline_source_sha256": pipeline_source_sha256(),
        "recipe_schema_sha256": RECIPE_SEQUENCE_SCHEMA_SHA256,
        "mask_runtime": {
            "runtime_id": runtime.runtime_id,
            "runtime_sha256": runtime.runtime_sha256,
            "config_sha256": runtime.config_sha256,
            "accepts_real_rgb_only": runtime.accepts_real_rgb_only,
        },
        "measurement": {
            "contract_id": "wp7-runtime-median-lab-delta-e76-v1",
            "contract_sha256": measurement_sha,
        },
        "decision": {
            "config": decision_config, "config_sha256": _hash(decision_config),
            "dp5_choice": "baseline", "dp5_passed_without_more_errors": False,
        },
        "explanation": {
            "method": "deterministic_guarded_template",
            "method_sha256": explanation_source_sha,
            "model_id": "none", "model_sha256": _hash({"model_id": "none"}),
            "prompt_sha256": _hash({
                "template": "Insufficient camera evidence; no keep or discard recommendation is made."
            }),
            "guard_sha256": _hash(guard_policy),
        },
        "store": {"schema_version": 1, "schema_sha256": store_sha},
        "validation_denominators": {
            "run_count": len(run_ids), "vial_count": len(vial_ids),
            "predicted_stage_count": len(rows), "measured_stage_count": len(rows),
            "history_supported_vial_count": 0, "explained_vial_count": 0,
        },
    }
    destination_path = Path(destination).expanduser().absolute()
    destination_path.mkdir(parents=True, exist_ok=False)
    try:
        validation_path = destination_path / "validation-results.json"
        validation_raw_sha = _write(validation_path, validation)
        freeze_path = destination_path / "freeze.json"
        freeze_value = freeze_validation(validation_path, freeze_path)
        freeze_raw_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
        receipt_path = destination_path / "freeze-preparation-receipt.json"
        receipt = {
            "schema_version": "wp7-cohort-freeze-preparation-v1",
            "artifact_type": "wp7_cohort_freeze_preparation",
            "status": "pass", "media_accessed": False,
            "inputs": {
                "checkpoint": {"path": str(checkpoint_path),
                               "sha256": expected_checkpoint_sha256},
                "replay": {"path": str(replay_path), "sha256": expected_replay_sha256},
                "review": {"path": str(review_path), "sha256": expected_review_sha256},
            },
            "outputs": {
                "validation-results.json": validation_raw_sha,
                "freeze.json": freeze_raw_sha,
            },
            "canonical_bindings": {
                "freeze_sha256": freeze_value["freeze_sha256"],
                "runtime_config_sha256": runtime.config_sha256,
                "decision_config_sha256": _hash(decision_config),
                "explanation_policy_sha256": _hash(guard_policy),
            },
        }
        receipt["publication_sha256"] = _hash(receipt)
        _write(receipt_path, receipt)
        return FreezePreparation(validation_path, freeze_path, receipt_path)
    except BaseException:
        shutil.rmtree(destination_path)
        raise


__all__ = [
    "CohortPreparation", "FreezePreparation", "prepare_run_test_inputs",
    "prepare_validation_freeze",
]

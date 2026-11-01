"""Metadata-only preparation for the accepted WP7 cohort-goldens route."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
import hashlib
import io
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
from ..perception.materializer import (
    CNN_CHECKPOINT_SHA256,
    CNN_REPLAY_SHA256,
    CNN_REVIEW_SHA256,
)
from .rgb_runtime import (
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


@dataclass(frozen=True)
class _AcceptedRunTestProfile:
    recipe_path: Path
    recipe_sha256: str
    composition_path: Path
    composition_sha256: str
    review_path: Path
    review_sha256: str
    run_paths: tuple[Path, Path]
    run_sha256s: tuple[str, str]
    video_sha256: str
    total_frame_count: int
    run_frame_counts: tuple[int, int]
    fill_count: int
    transition_count: int
    boundary_transition: tuple[int, int, int, int]
    run2_first_fill_frame_count: int
    first_transition: tuple[int, int]
    last_transition: tuple[int, int]
    cohorts: tuple[tuple[str, str, tuple[str, str, str]], ...]


@dataclass(frozen=True)
class _AcceptedRealCheckProfile:
    dataset_manifest_path: Path
    dataset_manifest_sha256: str
    dataset_inventory_path: Path
    dataset_inventory_sha256: str
    recipe_csv_path: Path
    recipe_csv_sha256: str
    transitions_csv_path: Path
    transitions_csv_sha256: str
    accepted_metadata_commit: str
    video_dataset_relative_path: str
    video_inventory_relative_path: str
    video_sha256: str
    video_byte_count: int
    total_frame_count: int
    fps: int
    width: int
    height: int
    vial_count: int
    fill_ranges: tuple[tuple[int, int], ...]


_ACCEPTED_ROOT = Path("/home/hackathon5/liamreporoot/outputs/pipeline-20261028")
_ACCEPTED_PRODUCT_ROOT = _ACCEPTED_ROOT / (
    "transport-01a18128/wp1-mixed-stitch-canonical-01a18128-20261030T060804Z/products"
)
_ACCEPTED_RUN_TEST_PROFILE = _AcceptedRunTestProfile(
    recipe_path=_ACCEPTED_ROOT / "wp7-onepass-preflight-01a18128/test-recipe-sequence.json",
    recipe_sha256="f789b26a01cb1297e8c6bd78b4316b0c877727cfbae3d13418acdc37229f4e16",
    composition_path=_ACCEPTED_ROOT / (
        "wp7-ordered-test-video-01a18128-attempt01/wp7-ordered-test-video.json"
    ),
    composition_sha256="0a88c47515af9fb0e389900f32504994914fe48bcc1e06bc392185382116b037",
    review_path=_ACCEPTED_ROOT / (
        "wp7-ordered-test-video-final-review-01a18128-attempt01/FINAL-REVIEW.json"
    ),
    review_sha256="ae4f9b2fdc3c6341a77dfa25044b673d7357155a98795ffa0f2df8b93ddec683",
    run_paths=(
        _ACCEPTED_PRODUCT_ROOT / "WP1-TEST-RUN-01.run.json",
        _ACCEPTED_PRODUCT_ROOT / "WP1-TEST-RUN-02.run.json",
    ),
    run_sha256s=(
        "c073c42b4ba73a563ed8d021c0b1148826fce4a5e3f9a9d3fb5bee575bc56e0d",
        "5d5a4daa0b267df78803a98e3f8d339dc543c73a9f35b11a2deec8cc63176d06",
    ),
    video_sha256="1021878c646f1019ac941ab645520446441d6f401ff664d05d5a4e64a682760b",
    total_frame_count=7322,
    run_frame_counts=(3574, 3748),
    fill_count=18,
    transition_count=19,
    boundary_transition=(9, 3561, 3587, 26),
    run2_first_fill_frame_count=322,
    first_transition=(0, 13),
    last_transition=(7309, 7322),
    cohorts=(
        ("WP1-TEST-RUN-01", "S0439", (
            "CASE_273915f5a10b2a7c", "CASE_284c9493d81e0724", "CASE_5af18c388c6021c9")),
        ("WP1-TEST-RUN-01", "S0440", (
            "CASE_91365fbcc3369a1c", "CASE_d52026e7df33b0cb", "CASE_ab6449c995c4b07d")),
        ("WP1-TEST-RUN-01", "S0441", (
            "CASE_645e5a76803161bd", "CASE_aed449240a936913", "CASE_76b13e1cb1e631c1")),
        ("WP1-TEST-RUN-02", "S0442", (
            "CASE_8eba6412a6e0f10e", "CASE_8c1aa64a892acf6f", "CASE_5e0fdb8dbe363a41")),
        ("WP1-TEST-RUN-02", "S0443", (
            "CASE_4bfc4ed18193b01c", "CASE_20a77ae6afdfc1fc", "CASE_f7046e01446adfe6")),
        ("WP1-TEST-RUN-02", "S0444", (
            "CASE_e79205fc1fce1ab5", "CASE_12807e933b7101ea", "CASE_188ed9e2e9701fc7")),
    ),
)


_REPOSITORY_ROOT = Path(__file__).absolute().parents[2]
_VIDEO5_FILL_RANGES = (
    (0, 808), (1386, 2210), (2752, 3616),
    (4148, 4841), (5373, 6094), (6630, 7393),
    (8540, 9242), (10105, 10830), (11740, 12428),
    (13043, 14184), (14754, 15849), (16463, 17597),
    (18292, 19692), (20386, 21780), (22485, 23926),
    (24530, 25681), (26267, 27409), (27973, 29098),
    (29685, 30929), (31511, 32768), (33326, 34600),
    (35387, 36619), (37392, 38698), (39495, 40797),
    (41534, 42711), (43540, 44620), (45430, 46548),
    (47385, 48035), (48751, 49400), (50138, 50815),
)
_ACCEPTED_REAL_CHECK_PROFILE = _AcceptedRealCheckProfile(
    dataset_manifest_path=_REPOSITORY_ROOT / "data/dataset_manifest.csv",
    dataset_manifest_sha256=(
        "4e3138a8f01b8418b2f910620393e2df950817d2ce24a0a5fe360e8d95fdfdcc"
    ),
    dataset_inventory_path=_REPOSITORY_ROOT / "data/dataset_inventory.csv",
    dataset_inventory_sha256=(
        "a97b19801ef82281e39a6507167cc8964564e07a6084da1518f6768f08f1c81f"
    ),
    recipe_csv_path=_REPOSITORY_ROOT / "data/video5_vial_recipes.csv",
    recipe_csv_sha256=(
        "4baf9246f8b34b803f8e3de4ef9915375311be83f5de91edf8cb249a78254f6d"
    ),
    transitions_csv_path=_REPOSITORY_ROOT / "data/video5_transitions.csv",
    transitions_csv_sha256=(
        "ecc4853249bd51419ed12e8107c412b5f419b5d4eb83214dff5db1f1e5a2006e"
    ),
    accepted_metadata_commit="c9a4a86e551f5d1b18676793cbdff04bcf6d38b8",
    video_dataset_relative_path=(
        "csi-vial-fill-dataset/vial_fill/videos/vial_fill_video_5.mp4"
    ),
    video_inventory_relative_path="vial_fill/videos/vial_fill_video_5.mp4",
    video_sha256=(
        "6f87cec1aeec3b2bae6dac639a317c077a3f175f27abd5fc8430d6953d9d05b9"
    ),
    video_byte_count=32178488,
    total_frame_count=50942,
    fps=30,
    width=640,
    height=480,
    vial_count=30,
    fill_ranges=_VIDEO5_FILL_RANGES,
)

# Derived from the frozen profile for the post-claim capture validator. The
# protected MP4 is deliberately represented by a dataset-relative string only.
VIDEO5_ACCEPTED_SOURCE_BINDINGS = {
    "accepted_metadata_commit": _ACCEPTED_REAL_CHECK_PROFILE.accepted_metadata_commit,
    "dataset_manifest": {
        "repository_relative_path": "data/dataset_manifest.csv",
        "raw_sha256": _ACCEPTED_REAL_CHECK_PROFILE.dataset_manifest_sha256,
    },
    "dataset_inventory": {
        "repository_relative_path": "data/dataset_inventory.csv",
        "raw_sha256": _ACCEPTED_REAL_CHECK_PROFILE.dataset_inventory_sha256,
    },
    "video5_vial_recipes": {
        "repository_relative_path": "data/video5_vial_recipes.csv",
        "raw_sha256": _ACCEPTED_REAL_CHECK_PROFILE.recipe_csv_sha256,
    },
    "video5_transitions": {
        "repository_relative_path": "data/video5_transitions.csv",
        "raw_sha256": _ACCEPTED_REAL_CHECK_PROFILE.transitions_csv_sha256,
    },
    "declared_video": {
        "dataset_relative_path": _ACCEPTED_REAL_CHECK_PROFILE.video_dataset_relative_path,
        "inventory_relative_path": _ACCEPTED_REAL_CHECK_PROFILE.video_inventory_relative_path,
        "byte_count": _ACCEPTED_REAL_CHECK_PROFILE.video_byte_count,
        "sha256": _ACCEPTED_REAL_CHECK_PROFILE.video_sha256,
        "frame_count": _ACCEPTED_REAL_CHECK_PROFILE.total_frame_count,
        "fps": _ACCEPTED_REAL_CHECK_PROFILE.fps,
        "width": _ACCEPTED_REAL_CHECK_PROFILE.width,
        "height": _ACCEPTED_REAL_CHECK_PROFILE.height,
        "vial_count": _ACCEPTED_REAL_CHECK_PROFILE.vial_count,
    },
    "fill_ranges": [list(bounds) for bounds in _VIDEO5_FILL_RANGES],
    "reference_rule": (
        "each_fill_start_frame_is_empty_vial_before_pour_including_frame_0"
    ),
}
VIDEO5_RUNTIME_CAPACITY_REQUIREMENTS = {
    "execution_mode": "capacity_bounded_in_memory_existing_runtime",
    "projected_peak_memory_bytes": 52 * 1024 ** 3,
    "minimum_available_ram_bytes": 80 * 1024 ** 3,
    "minimum_free_disk_bytes": 50 * 1024 ** 3,
    "measurement_timing": "immediately_before_run_real_check_claim",
}


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
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{role} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
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


def _validate_cohorts(
    sequence: Mapping[str, object], profile: _AcceptedRunTestProfile
) -> None:
    observed: list[tuple[str, str, tuple[str, str, str]]] = []
    for run in sequence["runs"]:
        assert isinstance(run, Mapping)
        vials = run["vials"]
        assert isinstance(vials, list)
        for start in range(0, len(vials), 3):
            rows = vials[start:start + 3]
            if len(rows) != 3 or any(not isinstance(row, Mapping) for row in rows):
                raise ValueError("accepted WP1 cohort triplets are malformed")
            recipe_id = rows[0].get("block_recipe_id")
            if (
                not isinstance(recipe_id, str)
                or [row.get("block_position") for row in rows] != [1, 2, 3]
                or any(row.get("block_recipe_id") != recipe_id for row in rows)
                or any(row.get("recipe_id") != recipe_id for row in rows)
            ):
                raise ValueError("accepted WP1 cohort triplets are malformed")
            case_tuple = tuple(str(row.get("case_id")) for row in rows)
            observed.append((str(run["run_id"]), recipe_id, case_tuple))
    if tuple(observed) != profile.cohorts:
        raise ValueError("accepted WP1 cohort identity or run grouping differs")


def _require_accepted_run_test_inputs(
    *,
    recipe_sequence: Path,
    expected_recipe_sequence_sha256: str,
    composition_receipt: Path,
    expected_composition_receipt_sha256: str,
    final_review: Path,
    expected_final_review_sha256: str,
    run_receipts: Sequence[Path],
    expected_run_receipt_sha256s: Sequence[str],
) -> None:
    profile = _ACCEPTED_RUN_TEST_PROFILE
    supplied_paths = (
        Path(recipe_sequence).expanduser().absolute(),
        Path(composition_receipt).expanduser().absolute(),
        Path(final_review).expanduser().absolute(),
    )
    if supplied_paths != (
        profile.recipe_path, profile.composition_path, profile.review_path
    ):
        raise ValueError("WP1 preparation path differs from the immutable accepted profile")
    if (
        expected_recipe_sequence_sha256 != profile.recipe_sha256
        or expected_composition_receipt_sha256 != profile.composition_sha256
        or expected_final_review_sha256 != profile.review_sha256
    ):
        raise ValueError("WP1 preparation SHA-256 differs from the immutable accepted profile")
    if (
        tuple(Path(path).expanduser().absolute() for path in run_receipts)
        != profile.run_paths
        or tuple(expected_run_receipt_sha256s) != profile.run_sha256s
    ):
        raise ValueError("WP1 run path/SHA order differs from the immutable accepted profile")


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


def _prepare_run_test_inputs_for_profile(
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
    profile: _AcceptedRunTestProfile,
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
    if len(case_ids) != profile.fill_count or any(
        len(run["vials"]) != 9 for run in sequence["runs"]
    ):
        raise ValueError("accepted WP1 geometry requires two nine-vial runs")
    _validate_cohorts(sequence, profile)
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
    prepared_recipe = (
        composition.get("provenance", {}).get("prepared_recipe")
        if isinstance(composition.get("provenance"), Mapping) else None
    )
    if (
        not isinstance(prepared_recipe, Mapping)
        or prepared_recipe.get("path") != str(sequence_path)
        or prepared_recipe.get("sha256") != expected_recipe_sequence_sha256
        or prepared_recipe.get("run_order") != run_ids
        or prepared_recipe.get("case_order") != case_ids
    ):
        raise ValueError("composition prepared-recipe binding differs")
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
    boundary_index, boundary_start, boundary_stop, boundary_count = (
        profile.boundary_transition
    )
    boundary = transitions[boundary_index] if len(transitions) > boundary_index else None
    if (
        video_sha != profile.video_sha256
        or frame_count != profile.total_frame_count
        or tuple(int(run["output_frame_count"]) for run in run_values)
        != profile.run_frame_counts
        or len(fills) != profile.fill_count
        or len(transitions) != profile.transition_count
        or not isinstance(boundary, Mapping)
        or (
            boundary.get("segment_index"), boundary.get("start_frame_index"),
            boundary.get("stop_frame_index"), boundary.get("frame_count"),
        ) != (boundary_index, boundary_start, boundary_stop, boundary_count)
        or run_values[1]["fill_clips"][0].get("frame_count")
        != profile.run2_first_fill_frame_count
        or (
            transitions[0]["start_frame_index"], transitions[0]["stop_frame_index"]
        ) != profile.first_transition
        or (
            transitions[-1]["start_frame_index"], transitions[-1]["stop_frame_index"]
        ) != profile.last_transition
    ):
        raise ValueError("accepted WP1 output geometry differs")
    review_recipe = (
        review.get("identity", {}).get("recipe")
        if isinstance(review.get("identity"), Mapping) else None
    )
    if (
        not isinstance(review_recipe, Mapping)
        or review_recipe.get("path") != str(sequence_path)
        or review_recipe.get("sha256") != expected_recipe_sequence_sha256
    ):
        raise ValueError("final review recipe binding differs")
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
    """Publish the one immutable accepted WP1 preparation bundle."""

    _require_accepted_run_test_inputs(
        recipe_sequence=recipe_sequence,
        expected_recipe_sequence_sha256=expected_recipe_sequence_sha256,
        composition_receipt=composition_receipt,
        expected_composition_receipt_sha256=expected_composition_receipt_sha256,
        final_review=final_review,
        expected_final_review_sha256=expected_final_review_sha256,
        run_receipts=run_receipts,
        expected_run_receipt_sha256s=expected_run_receipt_sha256s,
    )
    return _prepare_run_test_inputs_for_profile(
        recipe_sequence=recipe_sequence,
        expected_recipe_sequence_sha256=expected_recipe_sequence_sha256,
        composition_receipt=composition_receipt,
        expected_composition_receipt_sha256=expected_composition_receipt_sha256,
        final_review=final_review,
        expected_final_review_sha256=expected_final_review_sha256,
        run_receipts=run_receipts,
        expected_run_receipt_sha256s=expected_run_receipt_sha256s,
        destination=destination,
        profile=_ACCEPTED_RUN_TEST_PROFILE,
    )


def _bound_csv(
    path: Path, expected_sha256: str, role: str, expected_fields: tuple[str, ...]
) -> tuple[Path, list[dict[str, str]]]:
    bound_path, raw = _bound_bytes(path, expected_sha256, role)
    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"{role} fields are invalid")
        rows = [dict(row) for row in reader]
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"{role} must be strict UTF-8 CSV") from error
    if not rows or any(set(row) != set(expected_fields) for row in rows):
        raise ValueError(f"{role} rows are invalid")
    return bound_path, rows


def _repository_relative_source_binding(
    path: Path, expected_sha256: str, role: str,
) -> dict[str, str]:
    repository = _REPOSITORY_ROOT.absolute()
    candidate = Path(path).expanduser().absolute()
    try:
        repository_info = repository.lstat()
    except OSError as error:
        raise ValueError("repository root is invalid") from error
    if stat.S_ISLNK(repository_info.st_mode):
        raise ValueError("repository root must not be a symlink")
    try:
        relative = candidate.relative_to(repository)
    except ValueError as error:
        raise ValueError(f"{role} must be inside the repository") from error
    if (
        not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or repository.joinpath(*relative.parts) != candidate
    ):
        raise ValueError(f"{role} repository-relative path is invalid")
    current = repository
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise ValueError(f"{role} must be a regular non-symlink file") from error
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{role} must be a regular non-symlink file")
    return {
        "repository_relative_path": relative.as_posix(),
        "raw_sha256": expected_sha256,
    }


def _real_check_source_values(
    profile: _AcceptedRealCheckProfile,
) -> tuple[
    dict[str, object], list[dict[str, object]], list[dict[str, object]],
    list[dict[str, object]], dict[str, dict[str, object]],
]:
    manifest_path, manifest_rows = _bound_csv(
        profile.dataset_manifest_path, profile.dataset_manifest_sha256,
        "video5 dataset manifest", ("path", "bytes", "sha256"),
    )
    manifest_matches = [
        row for row in manifest_rows
        if row["path"] == profile.video_dataset_relative_path
    ]
    expected_manifest_row = {
        "path": profile.video_dataset_relative_path,
        "bytes": str(profile.video_byte_count),
        "sha256": profile.video_sha256,
    }
    if manifest_matches != [expected_manifest_row]:
        raise ValueError("video5 dataset manifest row identity is invalid")

    inventory_fields = (
        "camera", "video_id", "video_file", "video_bytes", "duration_s",
        "frames", "fps", "resolution", "codec", "n_stills", "first_still",
        "last_still", "still_span_s", "n_vials_reported", "labeled",
        "label_files", "recipe_ids", "license", "missing_unknown",
    )
    inventory_path, inventory_rows = _bound_csv(
        profile.dataset_inventory_path, profile.dataset_inventory_sha256,
        "video5 dataset inventory", inventory_fields,
    )
    inventory_matches = [
        row for row in inventory_rows
        if row["camera"] == "vial_fill" and row["video_id"] == "5"
    ]
    expected_inventory_row = {
        "camera": "vial_fill", "video_id": "5",
        "video_file": profile.video_inventory_relative_path,
        "video_bytes": str(profile.video_byte_count), "duration_s": "1698.067",
        "frames": str(profile.total_frame_count), "fps": str(profile.fps),
        "resolution": f"{profile.width}x{profile.height}", "codec": "h264",
        "n_stills": "845", "first_still": "2026-09-23 19:47:09.622176",
        "last_still": "2026-09-23 20:15:27.141712",
        "still_span_s": "1697.5", "n_vials_reported": "30", "labeled": "yes",
        "label_files": (
            "vial_fill/labels/video5_vial_recipes.csv;.json;"
            "video5_frame_labels.csv"
        ),
        "recipe_ids": "5-14", "license": "not stated",
        "missing_unknown": (
            "no defect labels (recipe/fill-window only); prior recipe before vial 1 "
            "unknown; timing measured: 1.2157 wall s per video s, so stills are "
            "missing for the last ~18% of each recording (see docs/video_timing.md); "
            "license not stated"
        ),
    }
    if inventory_matches != [expected_inventory_row]:
        raise ValueError("video5 dataset inventory row identity is invalid")

    recipe_fields = (
        "vial", "recipe_id", "type", "t100_ml", "t200_ml", "t400_ml",
        "total_ml", "dispense_order", "color_purpose", "start_time", "end_time",
        "start_s", "end_s", "duration_s", "start_frame", "end_frame",
    )
    recipe_path, recipe_rows = _bound_csv(
        profile.recipe_csv_path, profile.recipe_csv_sha256,
        "video5 vial recipes", recipe_fields,
    )
    transition_fields = (
        "video_id", "vial", "recipe_id", "dispense_order", "total_ml",
        "start_s", "end_s", "prev_vial", "prev_recipe_id", "recipe_changed",
        "position_in_run", "initial_batch", "prev_recipe_note",
    )
    transitions_path, transition_rows = _bound_csv(
        profile.transitions_csv_path, profile.transitions_csv_sha256,
        "video5 transitions", transition_fields,
    )
    if len(recipe_rows) != profile.vial_count or len(transition_rows) != profile.vial_count:
        raise ValueError("video5 recipe/transition vial denominator is invalid")

    sequence_vials: list[dict[str, object]] = []
    fill_clips: list[dict[str, object]] = []
    for index, (recipe, transition, bounds) in enumerate(
        zip(recipe_rows, transition_rows, profile.fill_ranges, strict=True), start=1
    ):
        expected_recipe = str(5 + (index - 1) // 3)
        expected_position = (index - 1) % 3 + 1
        start, stop = bounds
        if (
            recipe["vial"] != str(index)
            or recipe["recipe_id"] != expected_recipe
            or int(recipe["start_frame"]) != start
            or int(recipe["end_frame"]) + 1 != stop
            or transition["video_id"] != "5"
            or transition["vial"] != str(index)
            or transition["recipe_id"] != expected_recipe
            or transition["dispense_order"] != recipe["dispense_order"]
            or transition["total_ml"] != recipe["total_ml"]
            or transition["start_s"] != recipe["start_s"]
            or transition["end_s"] != recipe["end_s"]
            or transition["position_in_run"] != str(expected_position)
        ):
            raise ValueError("video5 recipe row/order/timing identity is invalid")
        previous_recipe = None if index == 1 else str(5 + (index - 2) // 3)
        if transition["prev_recipe_id"] != (
            "unknown" if previous_recipe is None else previous_recipe
        ):
            raise ValueError("video5 previous-recipe identity is invalid")
        volumes = {
            tank: (0 if recipe[field] == "" else int(recipe[field]))
            for tank, field in (
                ("t100", "t100_ml"), ("t200", "t200_ml"),
                ("t400", "t400_ml"),
            )
        }
        if sum(volumes.values()) != int(recipe["total_ml"]):
            raise ValueError("video5 programmed volume identity is invalid")
        order = [item.lower() for item in recipe["dispense_order"].split(">")]
        if set(order) != {tank for tank, amount in volumes.items() if amount > 0}:
            raise ValueError("video5 dispense order identity is invalid")
        case_id = f"VIDEO5-VIAL-{index:02d}"
        sequence_vials.append({
            "case_id": case_id, "vial_ordinal": index,
            "block_recipe_id": expected_recipe, "block_position": expected_position,
            "recipe_id": expected_recipe, "previous_recipe_id": previous_recipe,
            "programmed_tank_ml": volumes, "dispense_order": order,
        })
        fill_clips.append({"case_id": case_id, "frame_count": stop - start})

    sequence = {
        "schema_version": "carryover-recipe-sequence-v1",
        "split_role": "real_check",
        "split_version": "video5-cohort-goldens-v1",
        "runs": [{"run_id": "VIDEO5-RUN-01", "vials": sequence_vials}],
    }
    transition_ranges: list[dict[str, object]] = [{
        "segment_index": 0, "start_frame_index": 0, "stop_frame_index": 1,
        "frame_count": 1, "decoded_bgr_sha256": None,
    }]
    for index in range(1, profile.vial_count):
        start = profile.fill_ranges[index - 1][1]
        stop = profile.fill_ranges[index][0] + 1
        transition_ranges.append({
            "segment_index": index, "start_frame_index": start,
            "stop_frame_index": stop, "frame_count": stop - start,
            "decoded_bgr_sha256": None,
        })
    final_start = profile.fill_ranges[-1][1]
    transition_ranges.append({
        "segment_index": profile.vial_count, "start_frame_index": final_start,
        "stop_frame_index": profile.total_frame_count,
        "frame_count": profile.total_frame_count - final_start,
        "decoded_bgr_sha256": None,
    })
    if (
        len(transition_ranges) != profile.vial_count + 1
        or transition_ranges[-1]["stop_frame_index"] != profile.total_frame_count
    ):
        raise ValueError("video5 capture timing identity is invalid")
    sources = {
        "dataset_manifest": {
            **_repository_relative_source_binding(
                manifest_path, profile.dataset_manifest_sha256,
                "video5 dataset manifest",
            ),
            "row": expected_manifest_row,
        },
        "dataset_inventory": {
            **_repository_relative_source_binding(
                inventory_path, profile.dataset_inventory_sha256,
                "video5 dataset inventory",
            ),
            "row": expected_inventory_row,
        },
        "video5_vial_recipes": _repository_relative_source_binding(
            recipe_path, profile.recipe_csv_sha256, "video5 vial recipes",
        ),
        "video5_transitions": _repository_relative_source_binding(
            transitions_path, profile.transitions_csv_sha256, "video5 transitions",
        ),
    }
    return sequence, fill_clips, transition_ranges, sequence_vials, sources


def _prepare_real_check_inputs_for_profile(
    *, destination: Path, profile: _AcceptedRealCheckProfile
) -> CohortPreparation:
    sequence, fills, transitions, vials, sources = _real_check_source_values(profile)
    case_ids = [str(vial["case_id"]) for vial in vials]
    fill_windows = [
        {
            "fill_index": index, "case_id": case_ids[index],
            "start_frame_index": bounds[0], "stop_frame_index": bounds[1],
            "frame_count": bounds[1] - bounds[0],
        }
        for index, bounds in enumerate(profile.fill_ranges)
    ]
    timing_value = {
        "fill_clips": fills,
        "transition_ranges": [
            {key: row[key] for key in (
                "segment_index", "start_frame_index", "stop_frame_index", "frame_count"
            )}
            for row in transitions
        ],
        "fill_windows": fill_windows,
    }
    timing_sha = _hash(timing_value)
    route = dict(_ROUTE_DECISION)
    preflight = preflight_explanation_route(
        requested_mode="same_camera_cohort_goldens", expected_test_vials=18,
        expected_real_check_vials=30, route_decision=route,
        expected_route_decision_sha256=_hash(route),
    )
    destination_path = Path(destination).expanduser().absolute()
    destination_path.mkdir(parents=True, exist_ok=False)
    try:
        output_hashes: dict[str, str] = {}
        sequence_path = destination_path / "recipe-sequence.json"
        sequence_raw_sha = _write(sequence_path, sequence)
        sequence_canonical_sha = _hash(sequence)
        output_hashes[sequence_path.name] = sequence_raw_sha
        reference_contract = {
            "rule": "each_fill_start_frame_is_empty_vial_before_pour_including_frame_0",
            "reference_frame_indices": [start for start, _ in profile.fill_ranges],
            "evidence_status": "approved_metadata_assumption_not_physical_truth",
        }
        declared_video = {
            "dataset_relative_path": profile.video_dataset_relative_path,
            "byte_count": profile.video_byte_count, "sha256": profile.video_sha256,
            "frame_count": profile.total_frame_count, "fps": profile.fps,
            "width": profile.width, "height": profile.height, "path_accessed": False,
        }
        upstream_specs = (
            (
                "video5_dataset_manifest", "video5_dataset_manifest_v1",
                "upstream-01-dataset-manifest.json",
                {
                    "source_binding": sources["dataset_manifest"],
                    "declared_video": declared_video,
                },
            ),
            (
                "video5_recipe_metadata", "video5_recipe_metadata_v1",
                "upstream-02-recipe-metadata.json",
                {
                    "source_bindings": {
                        "video5_vial_recipes": sources["video5_vial_recipes"],
                        "video5_transitions": sources["video5_transitions"],
                    },
                    "recipe_sequence_raw_sha256": sequence_raw_sha,
                    "recipe_sequence_path": str(sequence_path),
                    "cohort_count": 10,
                    "reference_contract": reference_contract,
                },
            ),
            (
                "video5_timing_metadata", "video5_timing_metadata_v1",
                "upstream-03-timing-metadata.json",
                {
                    "source_binding": sources["dataset_inventory"],
                    "fill_count": 30, "transition_count": 31,
                    "reference_contract": reference_contract,
                },
            ),
            (
                "video5_transition_window_metadata",
                "video5_transition_window_metadata_v1",
                "upstream-04-transition-windows.json",
                {
                    "source_bindings": {
                        "video5_vial_recipes": sources["video5_vial_recipes"],
                        "video5_transitions": sources["video5_transitions"],
                    },
                    "fill_ranges": [list(bounds) for bounds in profile.fill_ranges],
                    "transition_ranges": timing_value["transition_ranges"],
                },
            ),
        )
        source_rows: list[dict[str, str]] = []
        for index, (role, artifact_type, upstream_name, extra) in enumerate(
            upstream_specs, start=1
        ):
            common = {
                "schema_version": 1, "artifact_type": artifact_type, "status": "pass",
                "accepted_metadata_commit": profile.accepted_metadata_commit,
                "video_sha256": profile.video_sha256, "case_ids": case_ids,
                "case_count": len(case_ids),
                "recipe_sequence_sha256": sequence_canonical_sha,
                "capture_timing_sha256": timing_sha,
                "decoded_frame_count": profile.total_frame_count,
            }
            upstream = {**common, **extra}
            upstream_path = destination_path / upstream_name
            upstream_sha = _write(upstream_path, upstream)
            output_hashes[upstream_name] = upstream_sha
            adapter = {
                "schema_version": "wp7-capture-source-adapter-v1",
                "artifact_type": "wp7_capture_source_adapter",
                "phase": "run-real-check", "role": role,
                "video_sha256": profile.video_sha256,
                "recipe_sequence_sha256": sequence_canonical_sha,
                "ordered_case_ids": case_ids, "capture_timing_sha256": timing_sha,
                "upstream_artifact_type": artifact_type,
                "upstream_path": str(upstream_path),
                "upstream_sha256": upstream_sha, "status": "pass",
            }
            adapter_name = f"source-{index:02d}-{role.removeprefix('video5_')}.json"
            adapter_path = destination_path / adapter_name
            adapter_sha = _write(adapter_path, adapter)
            output_hashes[adapter_name] = adapter_sha
            source_rows.append({
                "role": role, "path": str(adapter_path), "sha256": adapter_sha,
            })
        plan_path = destination_path / "capture-plan.json"
        plan = {
            "schema_version": "wp7-phase-capture-plan-v1",
            "artifact_type": "wp7_phase_capture_plan", "phase": "run-real-check",
            "capture_origin": "real_camera", "video_sha256": profile.video_sha256,
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
            "schema_version": "wp7-real-check-cohort-preparation-publication-v1",
            "artifact_type": "wp7_real_check_cohort_preparation_publication",
            "status": "pass", "access_policy": "metadata_only_no_media_access",
            "inputs": {
                "accepted_metadata_commit": profile.accepted_metadata_commit,
                "sources": sources,
                "recipe_sequence": {
                    "path": str(sequence_path), "raw_sha256": sequence_raw_sha,
                    "canonical_sha256": sequence_canonical_sha,
                },
                "declared_video": declared_video,
            },
            "reference_contract": reference_contract,
            "runtime_capacity_preflight": {
                "all_frames_raw_rgb_bytes": 46948147200,
                **VIDEO5_RUNTIME_CAPACITY_REQUIREMENTS,
                "capacity_evidence_status": (
                    "requirements_bound_not_measured_during_metadata_preparation"
                ),
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
        paths = tuple(destination_path / name for name in (*output_hashes,))
        return CohortPreparation(receipt, paths)
    except BaseException:
        shutil.rmtree(destination_path)
        raise


def prepare_real_check_inputs(*, destination: Path) -> CohortPreparation:
    """Publish the immutable metadata-only video-5 real-check preparation."""

    return _prepare_real_check_inputs_for_profile(
        destination=destination, profile=_ACCEPTED_REAL_CHECK_PROFILE
    )


def _available_memory_bytes() -> int:
    try:
        fields = {}
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                fields[key] = value.strip().split()
        available = fields["MemAvailable"]
        if len(available) != 2 or available[1] != "kB":
            raise ValueError
        result = int(available[0]) * 1024
    except (OSError, KeyError, ValueError) as error:
        raise ValueError("available RAM could not be measured fail-closed") from error
    if result <= 0:
        raise ValueError("available RAM could not be measured fail-closed")
    return result


def _free_disk_bytes(path: Path) -> int:
    candidate = Path(path).expanduser().absolute()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ValueError("destination free disk could not be measured fail-closed")
        candidate = parent
    try:
        result = int(shutil.disk_usage(candidate).free)
    except OSError as error:
        raise ValueError("destination free disk could not be measured fail-closed") from error
    if result <= 0:
        raise ValueError("destination free disk could not be measured fail-closed")
    return result


def require_video5_runtime_capacity(*, output_path: Path) -> dict[str, object]:
    """Fail closed unless the existing in-memory video-5 runtime has headroom."""

    observed_ram = _available_memory_bytes()
    observed_disk = _free_disk_bytes(Path(output_path).parent)
    required_ram = int(VIDEO5_RUNTIME_CAPACITY_REQUIREMENTS[
        "minimum_available_ram_bytes"
    ])
    required_disk = int(VIDEO5_RUNTIME_CAPACITY_REQUIREMENTS[
        "minimum_free_disk_bytes"
    ])
    if observed_ram < required_ram:
        raise ValueError(
            f"video5 runtime capacity preflight requires at least {required_ram} "
            f"available RAM bytes; observed {observed_ram}"
        )
    if observed_disk < required_disk:
        raise ValueError(
            f"video5 runtime capacity preflight requires at least {required_disk} "
            f"free destination disk bytes; observed {observed_disk}"
        )
    return {
        **VIDEO5_RUNTIME_CAPACITY_REQUIREMENTS,
        "observed_available_ram_bytes": observed_ram,
        "observed_free_disk_bytes": observed_disk,
        "status": "pass",
    }


def _source_hash(relative: str) -> str:
    path = Path(__file__).resolve().parents[3] / relative
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
    measurement_sha = _source_hash("src/fluidmix/measurement/vision_measurement.py")
    explanation_source_sha = _source_hash("src/fluidmix/pipeline/carryover_pipeline.py")
    store_sha = _source_hash("src/fluidmix/storage/sqlite/evidence_store.py")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
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
    "CohortPreparation", "FreezePreparation", "VIDEO5_ACCEPTED_SOURCE_BINDINGS",
    "VIDEO5_RUNTIME_CAPACITY_REQUIREMENTS", "prepare_real_check_inputs",
    "prepare_run_test_inputs", "prepare_validation_freeze",
    "require_video5_runtime_capacity",
]

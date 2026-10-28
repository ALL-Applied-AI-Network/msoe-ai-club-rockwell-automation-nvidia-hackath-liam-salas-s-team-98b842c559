"""Fail-closed CPU post-render decisions, acceptance, and packaging for WP1.

This module does not render or synthesize transport.  It only consumes frozen source
plans, immutable renderer artifacts, and separately reviewed transport products.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

from .full_transition_manifest import validate_full_transition_artifacts
from .physx_run_variation import validate_tank_colour_plan_overlay
from .run_sequence_dataset import validate_case_artifacts
from .vision_measurement import measure_image


_SHA256_CHARS = frozenset("0123456789abcdef")
_RUNTIME_FIELDS = (
    "gpu_model",
    "isaac_version",
    "kit_version",
    "stage_sha256",
    "renderer_script_sha256",
    "qualification_receipt_sha256",
    "nuisance_receipt_sha256",
    "python_runtime_sha256",
    "source_recipe_manifest_sha256",
    "source_split_spec_sha256",
)
_VISIBILITY_TARGET = "CASE_d52026e7df33b0cb"
_VISIBILITY_REFERENCE = "CASE_ab6449c995c4b07d"
_STAGES = ("t200", "t400", "t100")


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON semantics, which is not a file-byte hash."""

    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _read_object(path: Path, role: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{role} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return value


def _write_json_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _load_frozen_bundle(root: Path, repo_root: Path) -> dict[str, object]:
    source = _read_object(root / "full-transition-source-spec.json", "WP1 source spec")
    collection = _read_object(root / "full-transition-manifest.json", "WP1 collection")
    artifacts: dict[str, object] = {
        "full-transition-source-spec.json": source,
        "full-transition-manifest.json": collection,
    }
    leakage_path = root / str(collection.get("leakage_manifest_path", ""))
    if not leakage_path.is_file():
        raise ValueError("leakage_manifest is missing")
    artifacts["leakage-manifest.csv"] = leakage_path.read_text(encoding="utf-8")
    bindings = collection.get("run_bindings")
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise ValueError("collection run bindings are not the frozen two-run set")
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("collection run binding is malformed")
        for field, label in (
            ("render_plan_path", "render plan"),
            ("model_input_plan_path", "model input plan"),
            ("label_provenance_path", "label provenance"),
        ):
            relative = binding.get(field)
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError(f"{label} path is invalid")
            value = _read_object(root / relative, label)
            expected = binding.get(field.replace("_path", "_canonical_sha256"))
            if canonical_sha256(value) != expected:
                code = "label" if field == "label_provenance_path" else field
                raise ValueError(f"{code}_canonical_sha256 mismatch")
            artifacts[relative] = value
    return validate_full_transition_artifacts(artifacts, repo_root=repo_root)


def _plan_maps(
    artifacts: Mapping[str, object], binding: Mapping[str, object]
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    plan = artifacts[str(binding["render_plan_path"])]
    model = artifacts[str(binding["model_input_plan_path"])]
    if not isinstance(plan, dict) or not isinstance(model, dict):
        raise ValueError("bound child manifests are malformed")
    schedules = plan.get("execution_schedules")
    rows = model.get("vial_records")
    if not isinstance(schedules, list) or not isinstance(rows, list):
        raise ValueError("child manifest rows are malformed")
    schedule_by_case = {
        str(row["case_binding"]["case_id"]): row
        for row in schedules
        if isinstance(row, dict) and isinstance(row.get("case_binding"), dict)
    }
    model_by_case = {
        str(row["case_id"]): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("case_id"), str)
    }
    return plan, schedule_by_case, model_by_case


def _stage_pairs(receipt: Mapping[str, object], stages: Sequence[str]) -> list[dict[str, object]]:
    semantics = receipt.get("semantics")
    pairs = semantics.get("pairs") if isinstance(semantics, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("receipt-bound stage pairs are missing")
    result: list[dict[str, object]] = []
    positions: list[int] = []
    for stage in stages:
        matches = [
            (index, pair) for index, pair in enumerate(pairs)
            if isinstance(pair, dict) and pair.get("phase") == f"{stage}-stage"
        ]
        if len(matches) != 1:
            raise ValueError(f"receipt must contain exactly one {stage} stage pair")
        index, pair = matches[0]
        positions.append(index)
        if (
            pair.get("paired_capture_state_passed") is not True
            or pair.get("visible_stream_tanks") != []
            or pair.get("liquid_representation") != "vial_proxy"
        ):
            raise ValueError(f"{stage} stage pair lacks the frozen paired-state guards")
        result.append(pair)
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError("receipt stage phases are not in planned order")
    if len({pair.get("rgb_sha256") for pair in result}) != len(result):
        raise ValueError("ordered stage RGBs are not byte-unique")
    return result


def _absolute_locator(app_root: Path, pair: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for role in ("rgb", "mask"):
        relative = pair.get(f"{role}_path")
        digest = pair.get(f"{role}_sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or not _is_sha256(digest)
        ):
            raise ValueError(f"{role} locator is malformed")
        result[f"{role}_path"] = str((app_root / relative).resolve())
        result[f"{role}_sha256"] = digest
    return result


def _lab_distance(left: Sequence[object], right: Sequence[object]) -> float:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("CIE Lab descriptor is malformed")
    value = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    if not math.isfinite(value):
        raise ValueError("CIE76 is not finite")
    return value


def _visibility_payload(
    *, manifest_root: Path, run_root: Path, repo_root: Path
) -> dict[str, object]:
    artifacts = _load_frozen_bundle(manifest_root, repo_root)
    collection = artifacts["full-transition-manifest.json"]
    binding = collection["run_bindings"][0]
    plan, schedules, _ = _plan_maps(artifacts, binding)
    if list(binding["case_ids"])[4:6] != [_VISIBILITY_TARGET, _VISIBILITY_REFERENCE]:
        raise ValueError("visibility cases differ from the pre-registration")
    target_schedule = schedules[_VISIBILITY_TARGET]
    reference_schedule = schedules[_VISIBILITY_REFERENCE]
    target_binding = target_schedule["case_binding"]
    reference_binding = reference_schedule["case_binding"]
    if (
        target_binding.get("run_id") != "WP1-TEST-RUN-01"
        or target_binding.get("vial_ordinal") != 5
        or target_binding.get("current_recipe_id") != "S0440"
        or float(target_binding.get("residual_volume_ml", -1)) != 1.5
        or reference_binding.get("vial_ordinal") != 6
        or reference_binding.get("current_recipe_id") != "S0440"
        or float(reference_binding.get("residual_volume_ml", -1)) != 0.0
    ):
        raise ValueError("visibility case provenance differs from the pre-registration")
    stages = target_schedule["physical_sequence"]["dispense_order"]
    if stages != list(_STAGES) or reference_schedule["physical_sequence"]["dispense_order"] != stages:
        raise ValueError("visibility cases do not have the frozen three-stage order")

    warnings_by_case: dict[str, list[dict[str, object]]] = {}
    if (run_root / "shard-merge.json").is_file():
        warnings_by_case = _merge_warnings_and_validate(
            run_root, list(binding["case_ids"]), canonical_sha256(plan)
        )
    case_records: dict[str, dict[str, object]] = {}
    measured: dict[str, list[dict[str, object]]] = {}
    for role, case_id in (("target", _VISIBILITY_TARGET), ("reference", _VISIBILITY_REFERENCE)):
        case_root = run_root / "cases" / case_id
        evidence = validate_case_artifacts(
            case_root, height_warnings=warnings_by_case.get(case_id)
        )
        receipt_path = case_root / "app-output" / "receipt.json"
        receipt = _read_object(receipt_path, f"{role} renderer receipt")
        pairs = _stage_pairs(receipt, stages)
        values = []
        for stage, pair in zip(stages, pairs, strict=True):
            measurement = measure_image(
                f"{case_id}:{stage}", _absolute_locator(receipt_path.parent, pair)
            )
            if measurement.get("validity") != "valid":
                codes = measurement.get("limit_codes")
                raise ValueError(f"{role} {stage} mask evidence invalid: {codes}")
            values.append(measurement)
        measured[role] = values
        case_records[role] = {
            "case_id": case_id,
            "receipt_path": str(receipt_path.resolve()),
            "receipt_sha256": file_sha256(receipt_path),
            "artifact_validation": evidence,
            "rgb_sha256": [row["rgb_sha256"] for row in values],
            "mask_sha256": [row["mask_sha256"] for row in values],
        }
    if case_records["target"]["mask_sha256"] != case_records["reference"]["mask_sha256"]:
        raise ValueError("corresponding visibility masks have mismatched hashes")

    stage_rows = []
    for stage, left, right in zip(
        stages, measured["target"], measured["reference"], strict=True
    ):
        stage_rows.append(
            {
                "stage": stage,
                "target_lab_6dp": left["lab_6dp"],
                "reference_lab_6dp": right["lab_6dp"],
                "delta_e76": round(_lab_distance(left["lab_raw"], right["lab_raw"]), 6),
                "mask_sha256": left["mask_sha256"],
            }
        )
    maximum = max(row["delta_e76"] for row in stage_rows)
    passed = maximum >= 2.0
    payload = {
        "schema_version": 1,
        "artifact_type": "fluidmix-wp1-visibility-decision-v1",
        "status": "pass" if passed else "fallback_required",
        "decision": "original_1_5ml_visible" if passed else "add_fresh_3_0ml_fallback_cases",
        "threshold_delta_e76": 2.0,
        "aggregation": "maximum_same_stage_delta_e76",
        "maximum_delta_e76": maximum,
        "stage_measurements": stage_rows,
        "plan_binding": {
            "path": str((manifest_root / binding["render_plan_path"]).resolve()),
            "canonical_sha256": canonical_sha256(plan),
            "file_sha256": file_sha256(manifest_root / binding["render_plan_path"]),
        },
        "cases": case_records,
        "automatic_manifest_mutation": False,
        "excluded_from_model_inputs": True,
        "claim_boundary": "bounded synthetic pixel visibility only; not physical or real-camera measurability",
    }
    payload["decision_sha256"] = canonical_sha256(payload)
    return payload


def decide_wp1_visibility(
    *,
    manifest_root: str | Path,
    run_root: str | Path,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Measure the frozen run-1 vial-5/vial-6 DP3 pair and publish one decision."""

    root = Path(manifest_root).resolve()
    repository = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    payload = _visibility_payload(
        manifest_root=root, run_root=Path(run_root).resolve(), repo_root=repository
    )
    _write_json_new(Path(output_path), payload)
    return payload


def _merge_warnings_and_validate(
    run_root: Path, expected_case_ids: Sequence[str], plan_sha256: str
) -> dict[str, list[dict[str, object]]]:
    merge = _read_object(run_root / "shard-merge.json", "shard merge receipt")
    resume = merge.get("full_resume_summary")
    rows = resume.get("case_results") if isinstance(resume, dict) else None
    if (
        merge.get("schema_version") != 1
        or merge.get("artifact_type") != "physx_train_residual_shard_merge"
        or merge.get("status") != "pass"
        or merge.get("plan_sha256") != plan_sha256
        or merge.get("merged_case_count") != 9
        or merge.get("case_ids") != list(expected_case_ids)
        or not isinstance(resume, dict)
        or resume.get("status") != "pass"
        or resume.get("planned_case_count") != 9
        or resume.get("validated_completed_case_count") != 9
        or not isinstance(rows, list)
        or len(rows) != 9
    ):
        raise ValueError("merged nine-case root receipt is incomplete or mismatched")
    warnings: dict[str, list[dict[str, object]]] = {}
    observed = []
    for row in rows:
        evidence = row.get("evidence") if isinstance(row, dict) else None
        validation = evidence.get("artifact_validation") if isinstance(evidence, dict) else None
        case_id = row.get("case_id") if isinstance(row, dict) else None
        raw_warnings = validation.get("warnings") if isinstance(validation, dict) else None
        if (
            not isinstance(case_id, str)
            or row.get("outcome") not in {"passed", "recognized_completed"}
            or not isinstance(validation, dict)
            or validation.get("status") != "pass"
            or validation.get("failures") != []
            or not isinstance(raw_warnings, list)
            or any(not isinstance(warning, dict) for warning in raw_warnings)
        ):
            raise ValueError("merged case hard gates or warning provenance failed")
        observed.append(case_id)
        warnings[case_id] = deepcopy(raw_warnings)
    if observed != list(expected_case_ids):
        raise ValueError("merged case order differs from the frozen plan")
    return warnings


def _validate_runtime_profile(
    run_root: Path,
    *,
    plan: Mapping[str, object],
    plan_path: Path,
    case_ids: Sequence[str],
) -> tuple[dict[str, object], dict[str, object]]:
    profile = _read_object(run_root / "runtime-profile.json", "runtime profile receipt")
    if (
        profile.get("schema_version") != 1
        or profile.get("artifact_type") != "fluidmix-wp1-runtime-profile-v1"
        or profile.get("status") != "pass"
        or not isinstance(profile.get("execution_host"), str)
        or profile.get("plan_canonical_sha256") != canonical_sha256(plan)
        or profile.get("plan_file_sha256") != file_sha256(plan_path)
        or any(not _is_sha256(profile.get(field)) for field in _RUNTIME_FIELDS if field.endswith("sha256"))
    ):
        raise ValueError("runtime_profile receipt is malformed or plan-mismatched")
    source = plan.get("source_sha256")
    if (
        not isinstance(source, dict)
        or profile.get("source_recipe_manifest_sha256") != source.get("recipe_manifest")
        or profile.get("source_split_spec_sha256") != source.get("split_spec")
    ):
        raise ValueError("runtime_profile_mismatch: source hashes differ from the frozen plan")
    merge = _read_object(run_root / "shard-merge.json", "shard merge receipt")
    shards = merge.get("source_shards")
    hosts = {
        row.get("execution_host") for row in shards if isinstance(row, dict)
    } if isinstance(shards, list) else set()
    if hosts != {profile["execution_host"]}:
        raise ValueError("runtime_profile host differs from shard execution hosts")
    receipt_hashes = profile.get("case_receipt_sha256")
    if not isinstance(receipt_hashes, dict) or set(receipt_hashes) != set(case_ids):
        raise ValueError("runtime_profile case receipt set differs")
    for case_id in case_ids:
        receipt_path = run_root / "cases" / case_id / "app-output" / "receipt.json"
        receipt = _read_object(receipt_path, "renderer receipt")
        if receipt_hashes.get(case_id) != file_sha256(receipt_path):
            raise ValueError("runtime_profile case receipt hash differs")
        versions = receipt.get("versions")
        provenance = receipt.get("provenance")
        arguments = receipt.get("arguments")
        if (
            not isinstance(versions, dict)
            or versions.get("isaac") != profile.get("isaac_version")
            or versions.get("kit") != profile.get("kit_version")
            or not isinstance(provenance, dict)
            or provenance.get("requested_stage_sha256") != profile.get("stage_sha256")
            or provenance.get("script_sha256") != profile.get("renderer_script_sha256")
            or not isinstance(arguments, dict)
            or arguments.get("accepted_qualification_receipt_sha256")
            != profile.get("qualification_receipt_sha256")
            or arguments.get("nuisance_receipt_sha256")
            != profile.get("nuisance_receipt_sha256")
        ):
            raise ValueError("runtime_profile_mismatch: profile differs from a renderer receipt")
    comparable = {field: profile[field] for field in _RUNTIME_FIELDS}
    return profile, comparable


def _validate_transport(
    path: Path,
    *,
    binding: Mapping[str, object],
    source_fill_sha: Mapping[str, str],
) -> dict[str, object]:
    receipt = _read_object(path, "transport acceptance receipt")
    case_ids = list(binding["case_ids"])
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != "fluidmix-wp1-transport-acceptance-v1"
        or receipt.get("status") != "pass"
        or receipt.get("run_id") != binding.get("run_id")
        or receipt.get("case_ids") != case_ids
        or receipt.get("camera_transform_changed") is not False
        or receipt.get("qualified_fill_receipts_changed") is not False
        or receipt.get("review_status") != "accepted"
    ):
        raise ValueError("transport receipt identity, provenance, or review gate failed")
    segments = receipt.get("segments")
    expected_kinds = ["entry"] + ["between_fill"] * 8 + ["exit"]
    if not isinstance(segments, list) or len(segments) != 10:
        raise ValueError("transport segment count is not exactly 10")
    normalized_segments = []
    for ordinal, (row, kind) in enumerate(zip(segments, expected_kinds, strict=True), start=1):
        if not isinstance(row, dict):
            raise ValueError("transport segment row is malformed")
        center = row.get("incoming_center_error_px")
        background = row.get("background_max_abs_delta")
        seam = row.get("seam_mean_abs_difference")
        p99 = row.get("fill_interframe_p99")
        product = Path(str(row.get("product_path", "")))
        numeric_values = (center, background, seam, p99)
        if (
            row.get("ordinal") != ordinal
            or row.get("kind") != kind
            or row.get("passed") is not True
            or row.get("outgoing_fully_left_frame") is not True
            or isinstance(center, bool)
            or not isinstance(center, (int, float))
            or not all(math.isfinite(float(value)) for value in numeric_values if isinstance(value, (int, float)) and not isinstance(value, bool))
            or abs(float(center)) > 3.0
            or isinstance(background, bool)
            or not isinstance(background, (int, float))
            or float(background) < 0.0
            or float(background) > 2.0
            or isinstance(seam, bool)
            or not isinstance(seam, (int, float))
            or float(seam) < 0.0
            or isinstance(p99, bool)
            or not isinstance(p99, (int, float))
            or float(p99) < 0.0
            or float(seam) > float(p99)
            or not product.is_file()
            or file_sha256(product) != row.get("product_sha256")
        ):
            raise ValueError(f"transport segment {ordinal} gate or product failed")
        normalized_segments.append(deepcopy(row))
    products = receipt.get("vial_products")
    if not isinstance(products, list) or len(products) != 9:
        raise ValueError("transport vial product set is incomplete")
    normalized_products = []
    for ordinal, (row, case_id) in enumerate(zip(products, case_ids, strict=True), start=1):
        product = Path(str(row.get("product_path", ""))) if isinstance(row, dict) else Path()
        if (
            not isinstance(row, dict)
            or row.get("vial_ordinal") != ordinal
            or row.get("case_id") != case_id
            or row.get("source_fill_sha256") != source_fill_sha[case_id]
            or not product.is_file()
            or file_sha256(product) != row.get("product_sha256")
        ):
            raise ValueError(f"transport source_fill or vial product mismatch: {case_id}")
        normalized_products.append(deepcopy(row))
    run_product = receipt.get("run_product")
    run_path = Path(str(run_product.get("path", ""))) if isinstance(run_product, dict) else Path()
    if (
        not isinstance(run_product, dict)
        or not run_path.is_file()
        or file_sha256(run_path) != run_product.get("sha256")
    ):
        raise ValueError("transport run product failed its hash binding")
    return {
        "receipt_path": str(path.resolve()),
        "receipt_sha256": file_sha256(path),
        "segments": normalized_segments,
        "vial_products": normalized_products,
        "run_product": deepcopy(run_product),
    }


def _validate_auxiliary_receipts(
    manifest_root: Path,
    collection: Mapping[str, object],
) -> dict[str, object]:
    case_ids = [case for binding in collection["run_bindings"] for case in binding["case_ids"]]
    recipes = [recipe for binding in collection["run_bindings"] for recipe in binding["recipe_ids"]]
    leakage = _read_object(manifest_root / "leakage-check-receipt.json", "leakage receipt")
    if (
        leakage.get("artifact_type") != "fluidmix-wp1-leakage-check-v1"
        or leakage.get("status") != "pass"
        or leakage.get("split_version") != "inn-7-splits-v3"
        or leakage.get("case_ids") != case_ids
        or leakage.get("recipe_ids") != recipes
        or leakage.get("near_video5_count") != 0
        or leakage.get("leakage_failure_count") != 0
        or leakage.get("leakage_manifest_sha256")
        != file_sha256(manifest_root / "leakage-manifest.csv")
    ):
        raise ValueError("leakage receipt failed or differs from the frozen collection")
    boundary = _read_object(manifest_root / "input-boundary-receipt.json", "input boundary receipt")
    expected_model_hashes = [
        binding["model_input_plan_canonical_sha256"]
        for binding in collection["run_bindings"]
    ]
    if (
        boundary.get("artifact_type") != "fluidmix-wp1-input-boundary-check-v1"
        or boundary.get("status") != "pass"
        or boundary.get("forbidden_field_count") != 0
        or boundary.get("model_plan_canonical_sha256") != expected_model_hashes
    ):
        raise ValueError("input boundary receipt failed or was relabelled")
    return {
        "leakage_receipt_path": str((manifest_root / "leakage-check-receipt.json").resolve()),
        "leakage_receipt_sha256": file_sha256(manifest_root / "leakage-check-receipt.json"),
        "boundary_receipt_path": str((manifest_root / "input-boundary-receipt.json").resolve()),
        "boundary_receipt_sha256": file_sha256(manifest_root / "input-boundary-receipt.json"),
    }


def _validate_review(path: Path, run_shas: Sequence[str]) -> dict[str, object]:
    receipt = _read_object(path, "lead review receipt")
    video = Path(str(receipt.get("review_video_path", "")))
    if (
        receipt.get("artifact_type") != "fluidmix-wp1-review-acceptance-v1"
        or receipt.get("status") != "accepted"
        or receipt.get("reviewer_role") != "lead"
        or receipt.get("all_runs_viewed") is not True
        or receipt.get("visually_distinct_presets") is not True
        or receipt.get("run_product_sha256") != list(run_shas)
        or not video.is_file()
        or file_sha256(video) != receipt.get("review_video_sha256")
    ):
        raise ValueError("review artifact is absent, unaccepted, or hash-mismatched")
    return {
        "receipt_path": str(path.resolve()),
        "receipt_sha256": file_sha256(path),
        "video_path": str(video.resolve()),
        "video_sha256": file_sha256(video),
    }


def validate_wp1_acceptance(
    *,
    manifest_root: str | Path,
    run_roots: Sequence[str | Path],
    transport_receipts: Sequence[str | Path],
    review_receipt: str | Path,
    output_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate all mechanically checkable WP1 claims and always publish a result."""

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"acceptance output already exists: {destination}")
    repository = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "fluidmix-wp1-acceptance-v1",
        "status": "failed",
        "failures": [],
        "claim_boundary": "synthetic renderer and reviewed CPU transport evidence only; no plant-validity claim",
    }
    try:
        root = Path(manifest_root).resolve()
        artifacts = _load_frozen_bundle(root, repository)
        collection = artifacts["full-transition-manifest.json"]
        bindings = collection["run_bindings"]
        if len(run_roots) != 2:
            raise ValueError("case_set requires exactly two merged nine-case roots")
        if len(transport_receipts) != 2:
            raise ValueError("transport receipt set requires exactly two runs")
        auxiliary = _validate_auxiliary_receipts(root, collection)
        visibility = _visibility_payload(
            manifest_root=root,
            run_root=Path(run_roots[0]).resolve(),
            repo_root=repository,
        )
        if visibility["status"] != "pass":
            raise ValueError("visibility DP3 requires unrendered fallback cases")

        all_cases: list[dict[str, object]] = []
        runtime_comparables: list[dict[str, object]] = []
        runtime_profiles: list[dict[str, object]] = []
        source_fill_sha: dict[str, str] = {}
        label_counts: Counter[str] = Counter()
        recipe_ids: list[str] = []
        run_records: list[dict[str, object]] = []
        zero_measurements: dict[tuple[str, str], list[tuple[str, list[list[float]]]]] = defaultdict(list)
        for run_index, (raw_run_root, binding) in enumerate(
            zip(run_roots, bindings, strict=True), start=1
        ):
            run_root = Path(raw_run_root).resolve()
            plan_path = root / binding["render_plan_path"]
            plan, schedules, model_rows = _plan_maps(artifacts, binding)
            validate_tank_colour_plan_overlay(plan)
            expected_cases = list(binding["case_ids"])
            actual_cases = sorted(
                path.name for path in (run_root / "cases").glob("CASE_*") if path.is_dir()
            )
            if sorted(expected_cases) != actual_cases:
                raise ValueError("case_set differs from the frozen nine-case plan")
            warnings = _merge_warnings_and_validate(
                run_root, expected_cases, canonical_sha256(plan)
            )
            profile, comparable = _validate_runtime_profile(
                run_root, plan=plan, plan_path=plan_path, case_ids=expected_cases
            )
            runtime_profiles.append(profile)
            runtime_comparables.append(comparable)
            labels = artifacts[binding["label_provenance_path"]]["vial_labels"]
            label_by_case = {row["case_id"]: row for row in labels}
            ordered_recipes = []
            run_cases = []
            for ordinal, case_id in enumerate(expected_cases, start=1):
                schedule = schedules[case_id]
                model_row = model_rows[case_id]
                case_binding = schedule["case_binding"]
                recipe = case_binding["current_recipe_id"]
                ordered_recipes.append(recipe)
                label = label_by_case.get(case_id)
                residue = float(case_binding["residual_volume_ml"])
                expected_label = "keep" if residue == 0.0 else "discard"
                if (
                    not isinstance(label, dict)
                    or label.get("vial_ordinal") != ordinal
                    or label.get("expected_disposition") != expected_label
                    or label.get("prescribed_leading_residual_volume_ml") != residue
                ):
                    raise ValueError(f"label provenance was relabelled: {case_id}")
                label_counts[expected_label] += 1
                case_root = run_root / "cases" / case_id
                evidence = validate_case_artifacts(
                    case_root, height_warnings=warnings.get(case_id)
                )
                receipt_path = case_root / "app-output" / "receipt.json"
                receipt = _read_object(receipt_path, "renderer receipt")
                execution = receipt.get("execution_schedule")
                arguments = receipt.get("arguments")
                resolution = receipt.get("tank_colour_resolution")
                render = receipt.get("render")
                stages = schedule["physical_sequence"]["dispense_order"]
                pairs = _stage_pairs(receipt, stages)
                expected_stage_steps = [
                    pour["stage_capture_step"] for pour in schedule["pours"]
                ]
                if (
                    not isinstance(execution, dict)
                    or execution.get("case_id") != case_id
                    or execution.get("cohort_plan_sha256") != canonical_sha256(plan)
                    or execution.get("schedule_file_sha256") != file_sha256(plan_path)
                    or execution.get("source_run_spec_sha256") != plan["source_run_spec_sha256"]
                    or execution.get("schedule_sha256") != plan["execution_schedule_sha256_by_case"][case_id]
                    or execution.get("vial_record_sha256") != canonical_sha256(model_row)
                    or not isinstance(arguments, dict)
                    or arguments.get("video_fps") != 10.0
                    or resolution != plan["tank_colour_resolution"]
                    or not isinstance(render, dict)
                    or render.get("video_renderer") != "RaytracedLighting"
                    or render.get("stage_still_renderer") != "PathTracing"
                    or render.get("stage_frames_are_video_subsets") is not False
                    or render.get("stage_frame_paths")
                    != [pair["rgb_path"] for pair in pairs]
                    or render.get("stage_frame_sha256")
                    != [pair["rgb_sha256"] for pair in pairs]
                    or render.get("stage_frame_physics_steps") != expected_stage_steps
                ):
                    raise ValueError(f"canonical plan/receipt/stage/profile mismatch: {case_id}")
                fill_path = Path(str(evidence["source_video"]))
                fill_sha = str(evidence["source_video_sha256"])
                source_fill_sha[case_id] = fill_sha
                if residue == 0.0:
                    labs = []
                    app_root = receipt_path.parent
                    for stage, pair in zip(stages, pairs, strict=True):
                        measurement = measure_image(
                            f"{case_id}:{stage}", _absolute_locator(app_root, pair)
                        )
                        if measurement.get("validity") != "valid":
                            raise ValueError(f"stage mask measurement invalid: {case_id}:{stage}")
                        labs.append(measurement["lab_raw"])
                    zero_measurements[(binding["run_id"], recipe)].append((case_id, labs))
                case_record = {
                    "run_number": run_index,
                    "run_id": binding["run_id"],
                    "vial_ordinal": ordinal,
                    "case_id": case_id,
                    "recipe_id": recipe,
                    "disposition": expected_label.upper(),
                    "residue_ml": residue,
                    "source_fill_path": str(fill_path.resolve()),
                    "source_fill_sha256": fill_sha,
                    "receipt_path": str(receipt_path.resolve()),
                    "receipt_sha256": file_sha256(receipt_path),
                    "warnings": evidence["height_warnings"],
                    "renderer_warnings": evidence["renderer_warnings"],
                }
                all_cases.append(case_record)
                run_cases.append(case_record)
            expected_recipe_sequence = [recipe for recipe in binding["recipe_ids"] for _ in range(3)]
            if ordered_recipes != expected_recipe_sequence:
                raise ValueError("recipe case order is not exactly AAA-BBB-CCC")
            recipe_ids.extend(binding["recipe_ids"])
            run_records.append(
                {
                    "run_number": run_index,
                    "run_id": binding["run_id"],
                    "recipe_ids": binding["recipe_ids"],
                    "preset_name": binding["resolved_preset_name"],
                    "preset_hash": binding["resolved_preset_hash"],
                    "plan_path": str(plan_path.resolve()),
                    "plan_file_sha256": file_sha256(plan_path),
                    "plan_canonical_sha256": canonical_sha256(plan),
                    "cases": run_cases,
                }
            )
        if runtime_comparables[0] != runtime_comparables[1]:
            raise ValueError("runtime_profile_mismatch across the two host names")
        if runtime_comparables[0]["gpu_model"] != "NVIDIA L40S":
            raise ValueError("runtime_profile_mismatch: WP1 requires NVIDIA L40S")
        if len({profile["execution_host"] for profile in runtime_profiles}) != 2:
            raise ValueError("runtime_profile_mismatch: two distinct execution host names are required")
        if len(set(recipe_ids)) != 6 or label_counts != {"keep": 9, "discard": 9}:
            raise ValueError("counts, recipe isolation, or 9/9 label balance failed")
        sanity = []
        for (run_id, recipe), rows in zero_measurements.items():
            if len(rows) < 2:
                continue
            reference_id, reference_labs = rows[0]
            for case_id, labs in rows[1:]:
                deltas = [_lab_distance(left, right) for left, right in zip(reference_labs, labs, strict=True)]
                if max(deltas) > 2.0:
                    raise ValueError("within-block zero-residue stage sanity exceeded 2.0 DeltaE76")
                sanity.append({"run_id": run_id, "recipe_id": recipe, "left": reference_id, "right": case_id, "stage_delta_e76": [round(value, 6) for value in deltas]})

        transports = [
            _validate_transport(
                Path(path).resolve(), binding=binding, source_fill_sha=source_fill_sha
            )
            for path, binding in zip(transport_receipts, bindings, strict=True)
        ]
        for run, transport in zip(run_records, transports, strict=True):
            products = {row["case_id"]: row for row in transport["vial_products"]}
            for case in run["cases"]:
                case["transport_product_path"] = products[case["case_id"]]["product_path"]
                case["transport_product_sha256"] = products[case["case_id"]]["product_sha256"]
            run["transport"] = transport
        run_shas = [row["run_product"]["sha256"] for row in transports]
        review = _validate_review(Path(review_receipt).resolve(), run_shas)

        result.update(
            status="pass",
            failures=[],
            manifest_root=str(root),
            collection_manifest_sha256=file_sha256(root / "full-transition-manifest.json"),
            counts={"runs": 2, "cases": 18, "recipes": 6, "keep": 9, "discard": 9},
            runs=run_records,
            runtime={
                **runtime_comparables[0],
                "hosts": [profile["execution_host"] for profile in runtime_profiles],
                "profile_receipt_paths": [str((Path(run_root).resolve() / "runtime-profile.json")) for run_root in run_roots],
            },
            auxiliary_receipts=auxiliary,
            visibility=visibility,
            within_block_sanity=sanity,
            transport={"segment_count": 20, "receipts": transports},
            review=review,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        result["failures"] = [str(error)]
    result["acceptance_sha256"] = canonical_sha256(result)
    _write_json_new(destination, result)
    return result


def _copy_bound(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or file_sha256(source) != expected_sha256:
        raise ValueError(f"package source differs from acceptance binding: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if file_sha256(destination) != expected_sha256:
        raise RuntimeError(f"package copy verification failed: {destination}")


def package_wp1_test_videos(
    *, acceptance_path: str | Path, output_root: str | Path
) -> dict[str, object]:
    """Copy only accepted fill/transport/review products into the exact WP1 layout."""

    acceptance_source = Path(acceptance_path).resolve()
    acceptance = _read_object(acceptance_source, "WP1 acceptance")
    recorded_hash = acceptance.get("acceptance_sha256")
    without_hash = {key: value for key, value in acceptance.items() if key != "acceptance_sha256"}
    if acceptance.get("status") != "pass" or recorded_hash != canonical_sha256(without_hash):
        raise ValueError("WP1 package requires an intact passing acceptance receipt")
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"package output already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("package output parent must exist")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        test_root = staging / "videos" / "test"
        test_root.mkdir(parents=True)
        index_rows = []
        for run in acceptance["runs"]:
            number = int(run["run_number"])
            recipes = "-".join(run["recipe_ids"])
            run_name = f"test-run{number:02d}-{recipes}-{run['preset_name']}.mp4"
            run_product = run["transport"]["run_product"]
            _copy_bound(Path(run_product["path"]), test_root / run_name, run_product["sha256"])
            for segment in run["transport"]["segments"]:
                segment_name = f"segment-{int(segment['ordinal']):02d}-{segment['kind']}.mp4"
                _copy_bound(
                    Path(segment["product_path"]),
                    test_root / "review-inputs" / f"run-{number:02d}" / segment_name,
                    segment["product_sha256"],
                )
            for case in run["cases"]:
                vial_name = (
                    f"test-run{number:02d}-vial{int(case['vial_ordinal']):02d}-"
                    f"{case['disposition']}-{case['case_id']}.mp4"
                )
                _copy_bound(
                    Path(case["transport_product_path"]),
                    test_root / vial_name,
                    case["transport_product_sha256"],
                )
                fill_relative = f"review-inputs/source-fills/{case['case_id']}.mp4"
                _copy_bound(
                    Path(case["source_fill_path"]),
                    test_root / fill_relative,
                    case["source_fill_sha256"],
                )
                index_rows.append(
                    {
                        "run_id": run["run_id"],
                        "run_number": f"{number:02d}",
                        "vial_ordinal": f"{int(case['vial_ordinal']):02d}",
                        "case_id": case["case_id"],
                        "recipe_id": case["recipe_id"],
                        "disposition_metadata": case["disposition"],
                        "preset_metadata": run["preset_name"],
                        "vial_video": vial_name,
                        "run_video": run_name,
                        "accepted_fill_review_input": fill_relative,
                        "accepted_fill_sha256": case["source_fill_sha256"],
                        "transport_product_sha256": case["transport_product_sha256"],
                    }
                )
        review = acceptance["review"]
        _copy_bound(
            Path(review["video_path"]), test_root / "wp1-review.mp4", review["video_sha256"]
        )
        _copy_bound(
            Path(review["receipt_path"]),
            test_root / "review-inputs" / "lead-review-receipt.json",
            review["receipt_sha256"],
        )
        _copy_bound(
            acceptance_source,
            test_root / "review-inputs" / "wp1-acceptance.json",
            file_sha256(acceptance_source),
        )
        with (test_root / "INDEX.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
            writer.writeheader()
            writer.writerows(index_rows)
        (test_root / "README.md").write_text(
            "# WP1 synthetic test videos\n\n"
            "This directory contains 2 reviewed synthetic runs and 18 labeled audit clips. "
            "Disposition and preset columns in `INDEX.csv` are metadata, not model inputs. "
            "`review-inputs/` preserves hash-bound accepted fill clips, transport segments, "
            "and review receipts. The package makes no physical or plant-validity claim.\n",
            encoding="utf-8",
        )
        paths = sorted(path for path in staging.rglob("*") if path.is_file())
        sums = staging / "SHA256SUMS"
        with sums.open("x", encoding="utf-8") as stream:
            for path in paths:
                stream.write(f"{file_sha256(path)}  {path.relative_to(staging).as_posix()}\n")
        receipt = {
            "schema_version": 1,
            "artifact_type": "fluidmix-wp1-test-video-package-v1",
            "status": "pass",
            "acceptance_path": str(acceptance_source),
            "acceptance_file_sha256": file_sha256(acceptance_source),
            "run_video_count": 2,
            "vial_video_count": 18,
            "transport_segment_count": 20,
            "accepted_fill_input_count": 18,
            "sha256sums_sha256": file_sha256(sums),
            "transport_gates_weakened": False,
            "qualified_fill_receipts_changed": False,
        }
        _write_json_new(staging / "package-receipt.json", receipt)
        os.rename(staging, destination)
        return receipt
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = [
    "canonical_sha256",
    "decide_wp1_visibility",
    "file_sha256",
    "package_wp1_test_videos",
    "validate_wp1_acceptance",
]

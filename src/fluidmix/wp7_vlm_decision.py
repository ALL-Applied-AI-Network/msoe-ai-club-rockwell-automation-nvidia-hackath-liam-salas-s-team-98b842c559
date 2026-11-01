"""CPU-only image-grounded WP7 vial-1 decision contracts.

It consumes hash-bound camera/CNN observations and provides a sequential adapter
over the existing pinned stock-model runtime. Importing the module never loads a model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from fluidmix.model_bakeoff_runner import MODEL_SPECS


class ContractError(ValueError):
    """A WP7 decision artifact violates the frozen contract."""


PHASES = ("early", "mid", "end")
DISPOSITIONS = {"keep", "discard", "insufficient_evidence"}
STAGE_RELIABILITY = ("clear_consistent", "weak", "low", "conflicting")
REFERENCE_RELATIONS = ("keep_range", "deadband", "discard_range")
STREAM_TANK_CONSISTENCY = ("consistent", "inconsistent")
PREVIOUS_RECIPE_DIRECTIONS = (
    "toward_previous_recipe", "not_toward_previous_recipe", "not_assessable",
)
AGGREGATION = {"threshold": 0.5, "minimum_valid_sources": 2, "weighting": "equal"}
MODELS = tuple(
    {"repository": repository, "revision": MODEL_SPECS[repository].revision}
    for repository in (
        "Qwen/Qwen3-VL-8B-Instruct", "nvidia/Cosmos-Reason2-8B",
        "aacudad/AnomalyThink-Qwen3-VL-8B-KCR",
    )
)
SYSTEM_PROMPT = """You assess a sequential fluid-mixing fill from evidence in one supplied bundle.
T100, T200, and T400 are dosed in the listed order; T300 is out of service.
Each addition can mix incompletely or remain stratified. During a recipe changeover, a leading slug
remaining in the shared dispense path can shift vial 1 toward the previous recipe; vials 2 and 3
are flushed same-recipe references. This is a comparison pattern, not permission to claim
a cause.

Compare early, mid, and end images like-for-like: same stage, fill level, frozen vial ROI, fixed
camera metadata, viewpoint, lighting, and phase. Meniscus shape, bubbles, foam, splashing,
settling, and the stable end-of-fill window can change appearance; that stable end window is the
most trustworthy visual window. Reflections, specular highlights,
refraction, vial curvature, clamp occlusion, motion blur, and lighting drift are possible camera
effects. Read fixed resolution, frame rate, exposure, white balance, viewpoint, and vial ROI from
the bundle. Treat conflicting weak evidence as insufficient evidence.

CIELAB L* is lightness, a* is green-to-red, and b* is blue-to-yellow. dE76 is Euclidean distance
in those three channels. The frozen rule is keep only when both same-stage reference distances
are at most 1.0, discard only when both are at least 2.0, and otherwise insufficient evidence.
MAD and IQR describe measurement spread and therefore trust, not a new threshold. Use each
stream's median Lab and spread with its mL, tank, and order. Check tracker
tank and step identity, measured-versus-expected fill agreement, and the exact supplied
tank_cluster_consistent boolean. Report stream-tank consistency only as consistent when that
boolean is true and inconsistent when it is false. Check matching fill levels. Assess the
previous-recipe direction only when the bundle says a previous-recipe colour reference is
available; otherwise report not_assessable and make no directional claim in prose.

Confidence at or above 0.9 requires every stage to be clear_consistent and the image evidence
to agree. If any stage is weak, low, or conflicting, confidence must be at most 0.8. Below 0.6 must use
insufficient_evidence. Give every observation its exact supplied bundle IDs. Cite supplied
bundle IDs only. Do not add unsupported causes or process advice."""
RESPONSE_FIELD_ORDER = (
    "observations", "per_stage_comparison", "disposition",
    "disposition_confidence", "explanation", "evidence_citations",
    "image_citations",
)
RESPONSE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": list(RESPONSE_FIELD_ORDER),
    "properties": {
        "observations": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["text", "bundle_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "bundle_ids": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "per_stage_comparison": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "stage_id", "phase_and_fill_match", "vial1_vs_vial2",
                    "vial1_vs_vial3", "deterministic_stage_disposition",
                    "stream_tank_consistency",
                    "previous_recipe_direction", "reliability", "bundle_ids",
                ],
                "properties": {
                    "stage_id": {"type": "string", "minLength": 1},
                    "phase_and_fill_match": {"type": "string", "minLength": 1},
                    "vial1_vs_vial2": {"enum": list(REFERENCE_RELATIONS)},
                    "vial1_vs_vial3": {"enum": list(REFERENCE_RELATIONS)},
                    "deterministic_stage_disposition": {
                        "enum": sorted(DISPOSITIONS),
                    },
                    "stream_tank_consistency": {
                        "enum": list(STREAM_TANK_CONSISTENCY),
                    },
                    "previous_recipe_direction": {
                        "enum": list(PREVIOUS_RECIPE_DIRECTIONS),
                    },
                    "reliability": {
                        "enum": list(STAGE_RELIABILITY),
                    },
                    "bundle_ids": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "disposition": {"enum": sorted(DISPOSITIONS)},
        "disposition_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "explanation": {"type": "string"},
        "evidence_citations": {"type": "array", "items": {"type": "string"}},
        "image_citations": {"type": "array", "items": {"type": "string"}},
    },
}
USER_PROMPT_TEMPLATE = """Apply the domain contract to the evidence bundle below. Follow its
sequential mixture order, compare matching stage/phase/fill evidence, use MAD/IQR to qualify
trust, check tracker measured-versus-expected agreement and bind stream-tank consistency to
each stage's exact tank_cluster_consistent boolean. Compare vial 1 with both flushed references.
For previous-recipe direction, obey previous_recipe_direction_support: when unavailable, emit
not_assessable for every stage and do not mention the previous recipe in observations or the
explanation except to state truthfully that support is unavailable, direction is not assessable,
or no directional claim is made. Each stage comparison's bundle_ids may contain only IDs
belonging to that exact stage. Use the exact 1.0/2.0
dE76 rules and confidence rubric. Report each reference comparison only as keep_range (at most
1.0), deadband (above 1.0 and below 2.0), or discard_range (at least 2.0), and copy the
bundle's deterministic stage disposition. These structured outcomes are authoritative;
observations, physical comparison text, and the explanation must not contradict or restate
the threshold math. When signals conflict or support is weak, return
insufficient_evidence. Every observation must contain text plus exact supplied bundle_ids.
Set each stage reliability to clear_consistent only if its image evidence is clear and agrees;
otherwise use weak, low, or conflicting. If ANY stage is not clear_consistent,
disposition_confidence MUST be at most 0.8. Confidence at or above 0.9 is allowed only if EVERY
stage is clear_consistent and the image evidence agrees.
Cite bundle IDs only and give no unsupported cause or process advice.
Response schema (property order is required):
{schema}
Evidence bundle:
{bundle}"""
_FORBIDDEN = (
    "simulator", "residue", "palette", "oracle", "semantic_mask", "particles",
    "truth", "test_label", "expected_lab", "beer_lambert", "absorbance",
)
_PATH = re.compile(r"[\\/]|(?:^|[^.])\.\.(?:[^.]|$)")
_PREVIOUS_RECIPE_TEXT = re.compile(
    r"(?:previous|prior|preceding|predecessor)[\s_-]+recipe", re.I
)
_DIRECTIONAL_CLAIM_TEXT = re.compile(
    r"\b(?:toward(?:s)?|closer|align(?:ed|ment|s|ing)?|"
    r"shift(?:s|ed|ing)?|move(?:s|d|ment|ing)?)\b",
    re.I,
)
_NO_PREVIOUS_RECIPE_DIRECTION_SUPPORT = {
    "available": False,
    "reason": "previous_recipe_color_reference_unavailable",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
TRACKER_CLAIM_BOUNDARY = "camera_twin_geometry_equivalent_not_physical_ground_truth"
DETERMINISTIC_CONFIG_SHA256 = (
    "6145a4da5c452ff4c8e8d4c6018ffeb5009410501f2a7438efd867317962057e"
)
DETERMINISTIC_POLICY = {
    "distance_metric": "CIE76",
    "keep": {"comparison": "both_lte", "threshold": 1.0},
    "discard": {"comparison": "both_gte", "threshold": 2.0},
    "deadband_disposition": "insufficient_evidence",
    "deadband_limit_code": "same_cohort_threshold_deadband",
    "overall_discard": "any_stage_discard",
    "overall_keep": "all_stages_keep",
    "overall_other": "insufficient_evidence",
}
MODEL_REASONING_MODES = {
    "Qwen/Qwen3-VL-8B-Instruct": "schema_json",
    "nvidia/Cosmos-Reason2-8B": "native_think_answer",
    "aacudad/AnomalyThink-Qwen3-VL-8B-KCR": "native_think_answer",
}
GENERATION_CONFIG = {
    "do_sample": False, "max_new_tokens": 10000, "batch_size": 1, "seed": 17,
    "torch_deterministic_algorithms": True, "cudnn_benchmark": False,
    "cudnn_deterministic": True, "cublas_workspace_config": ":4096:8",
    "processor_pixels": {"min_pixels": 1, "max_pixels": 16777216},
    "per_model_fallbacks": {repository: None for repository in MODEL_REASONING_MODES},
}
_TRACKER_RECIPE_KEYS = (
    "fill_step", "tank_id", "tank_number", "stream_step_index",
    "expected_dispense_ml", "expected_cumulative_ml",
)


@dataclass(frozen=True)
class BoundImageLocator:
    path: Path
    sha256: str
    source_full_frame_sha256: str
    crop_pixel_sha256: str
    crop_box_xyxy: tuple[int, int, int, int]
    source_native_dimensions: tuple[int, int]
    crop_native_dimensions: tuple[int, int]


@dataclass(frozen=True)
class BuiltBundle:
    prompt_payload: dict[str, Any]
    image_locators: dict[str, BoundImageLocator]
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class RuntimeContext:
    built: BuiltBundle
    freeze_sha256: str
    binding: dict[str, Any]
    production: bool


def _configure_runtime_determinism(runtime: object) -> None:
    torch_module = getattr(runtime, "torch_module", None)
    if torch_module is None:
        return
    torch_module.manual_seed(GENERATION_CONFIG["seed"])
    if getattr(torch_module.cuda, "is_available", lambda: False)():
        torch_module.cuda.manual_seed_all(GENERATION_CONFIG["seed"])
    torch_module.use_deterministic_algorithms(True)
    cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ContractError("artifact contains nonfinite or non-JSON value") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _opaque_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not any(token in value for token in ("/", "\\", ".."))
    )


def _finite(value: object, role: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{role} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ContractError(f"{role} is nonfinite or outside range")
    return result


def _lab(value: object, role: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ContractError(f"{role} must have three channels")
    return [_finite(item, role) for item in value]


def _measurement(value: Mapping[str, object], *, target: bool) -> dict[str, object]:
    for field in ("median_lab", "mad_lab", "iqr_lab"):
        _lab(value[field], field)
    if target:
        for field in ("delta_e76_vial2", "delta_e76_vial3"):
            _finite(value[field], field, 0.0)
        if value["deterministic_disposition"] not in DISPOSITIONS:
            raise ContractError("deterministic disposition is invalid")
    if not isinstance(value["evidence_id"], str) or not value["evidence_id"]:
        raise ContractError("evidence ID is invalid")
    if not isinstance(value["limit_codes"], list) or any(
        not isinstance(item, str) or not item for item in value["limit_codes"]
    ):
        raise ContractError("deterministic limit codes are invalid")
    result = dict(value)
    if not target:
        for field in (
            "deterministic_disposition", "delta_e76_vial2", "delta_e76_vial3"
        ):
            result.pop(field, None)
    return result


def _delta_e76(left: Sequence[object], right: Sequence[object]) -> float:
    left_lab, right_lab = _lab(left, "left median Lab"), _lab(right, "right median Lab")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left_lab, right_lab, strict=True)))


def _deterministic_stage_call(delta_e76_vial2: object, delta_e76_vial3: object) -> tuple[str, list[str]]:
    distances = (
        _finite(delta_e76_vial2, "deterministic delta E76", 0.0),
        _finite(delta_e76_vial3, "deterministic delta E76", 0.0),
    )
    if all(distance <= 1.0 for distance in distances):
        return "keep", []
    if all(distance >= 2.0 for distance in distances):
        return "discard", []
    return "insufficient_evidence", ["same_cohort_threshold_deadband"]


def _reference_relation(delta_e76: object) -> str:
    distance = _finite(delta_e76, "reference delta E76", 0.0)
    if distance <= 1.0:
        return "keep_range"
    if distance >= 2.0:
        return "discard_range"
    return "deadband"


def _validate_deterministic_receipt(
    value: object, *, cohort_id: object, target_vial_id: object,
    target_rows: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("deterministic decision receipt is required")
    fields = {
        "schema_version", "artifact_type", "cohort_id", "target_vial_id",
        "disposition", "limit_codes", "config_sha256", "source_artifact_sha256",
        "stages", "receipt_sha256",
    }
    if set(value) != fields:
        raise ContractError("deterministic decision receipt fields are not closed")
    if (
        value.get("schema_version") != "wp7-deterministic-decision-receipt-v1"
        or value.get("artifact_type") != "wp7_deterministic_decision_receipt"
        or value.get("cohort_id") != cohort_id
        or value.get("target_vial_id") != target_vial_id
        or value.get("disposition") not in DISPOSITIONS
    ):
        raise ContractError("deterministic decision receipt identity is invalid")
    for field in ("config_sha256", "source_artifact_sha256", "receipt_sha256"):
        if not isinstance(value.get(field), str) or _SHA.fullmatch(value[field]) is None:
            raise ContractError("deterministic decision receipt hashes are invalid")
    observed = _digest({key: item for key, item in value.items() if key != "receipt_sha256"})
    if observed != value["receipt_sha256"]:
        raise ContractError("deterministic decision receipt SHA mismatch")
    if value["config_sha256"] != DETERMINISTIC_CONFIG_SHA256:
        raise ContractError("deterministic config is outside the frozen policy")
    if not isinstance(value.get("limit_codes"), list) or any(
        not isinstance(item, str) or not item for item in value["limit_codes"]
    ):
        raise ContractError("deterministic decision receipt limit codes are invalid")
    stages = value.get("stages")
    if not isinstance(stages, list) or len(stages) != len(target_rows):
        raise ContractError("deterministic decision receipt stages differ")
    stage_fields = {
        "stage_id", "evidence_id", "disposition", "limit_codes",
        "delta_e76_vial2", "delta_e76_vial3",
    }
    for receipt_stage, target in zip(stages, target_rows, strict=True):
        if not isinstance(receipt_stage, Mapping) or set(receipt_stage) != stage_fields:
            raise ContractError("deterministic decision receipt stage fields are not closed")
        expected = {
            "stage_id": target["stage_id"],
            "evidence_id": target["evidence_id"],
            "disposition": target["deterministic_disposition"],
            "limit_codes": target["limit_codes"],
            "delta_e76_vial2": target["delta_e76_vial2"],
            "delta_e76_vial3": target["delta_e76_vial3"],
        }
        if dict(receipt_stage) != expected:
            raise ContractError("deterministic decision receipt does not bind target evidence")
        expected_call, expected_limits = _deterministic_stage_call(
            receipt_stage["delta_e76_vial2"], receipt_stage["delta_e76_vial3"]
        )
        if (
            receipt_stage["disposition"] != expected_call
            or receipt_stage["limit_codes"] != expected_limits
        ):
            raise ContractError("deterministic rule call is outside the frozen policy")
    stage_dispositions = [stage["disposition"] for stage in stages]
    expected_overall = (
        "discard" if "discard" in stage_dispositions
        else "keep" if stage_dispositions and set(stage_dispositions) == {"keep"}
        else "insufficient_evidence"
    )
    expected_limits = list(dict.fromkeys(
        code for stage in stages for code in stage["limit_codes"]
    ))
    if (
        value["disposition"] != expected_overall
        or value["limit_codes"] != expected_limits
    ):
        raise ContractError("deterministic overall rule call is outside the frozen policy")
    return dict(value)


def validate_tracker_receipt(value: object) -> dict[str, Any]:
    """Delegate to the dedicated tracker's exact closed-schema validator."""
    from fluidmix.wp7_fill_step_tracker import (
        FillStepTrackerError,
        validate_fill_step_receipt,
    )

    try:
        checked = validate_fill_step_receipt(value)
    except (FillStepTrackerError, TypeError, ValueError) as exc:
        raise ContractError(f"tracker receipt invalid: {exc}") from exc
    for stage in checked["stages"]:
        images = stage["images"]
        if len(set(images.values())) != 3:
            raise ContractError("tracker early/mid/end images must be distinct")
        if stage["tank_cluster_consistent"] is not True:
            raise ContractError("tracker tank cluster is inconsistent")
    return checked


def _validate_tracker_cohort_receipts(receipts: object) -> list[dict[str, Any]]:
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)) or len(receipts) != 3:
        raise ContractError("exactly three tracker receipts are required")
    checked = [validate_tracker_receipt(item) for item in receipts]
    if len({item["cohort_id"] for item in checked}) != 1:
        raise ContractError("tracker/cohort identity mismatch")
    if len({item["vial_id"] for item in checked}) != 3:
        raise ContractError("tracker vial identities must be unique")
    for field in ("calibration_sha256", "recipe_artifact_sha256", "claim_boundary"):
        if len({item[field] for item in checked}) != 1:
            raise ContractError(f"tracker {field} differs within cohort")
    if len({item["source_artifact_sha256"] for item in checked}) != 3:
        raise ContractError("tracker source artifacts must be distinct per vial")
    target_recipe = [
        tuple(stage[key] for key in _TRACKER_RECIPE_KEYS)
        for stage in checked[0]["stages"]
    ]
    if any(
        [
            tuple(stage[key] for key in _TRACKER_RECIPE_KEYS)
            for stage in receipt["stages"]
        ] != target_recipe
        for receipt in checked[1:]
    ):
        raise ContractError("tracker recipe state differs within cohort")
    return checked


def _tracker_cohort(receipts: object, cohort_id: object, vials: list[object]) -> list[dict[str, Any]]:
    checked = _validate_tracker_cohort_receipts(receipts)
    if checked[0]["cohort_id"] != cohort_id:
        raise ContractError("tracker/cohort identity mismatch")
    expected_vial_ids = [item.get("vial_id") for item in sorted(vials, key=lambda row: row["vial_position"])]
    if [item["vial_id"] for item in checked] != expected_vial_ids:
        raise ContractError("tracker vial order/identity mismatch")
    return checked


def _scan(value: object, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN):
                raise ContractError(f"forbidden field at {path}/{key}")
            _scan(item, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan(item, f"{path}/{index}")
    elif isinstance(value, str) and (_PATH.search(value) or value.startswith(("file:", "~"))):
        raise ContractError(f"path-like prompt value at {path}")


def _camera(value: object, production: bool) -> dict[str, object]:
    fields = ("resolution", "frame_rate_hz", "exposure", "white_balance", "viewpoint_id", "vial_roi")
    if not isinstance(value, Mapping) or not set(value).issubset(fields):
        raise ContractError("camera metadata fields are invalid")
    result = {field: value.get(field, "unavailable") for field in fields}
    result = {key: "unavailable" if item is None else item for key, item in result.items()}
    if result["frame_rate_hz"] != "unavailable":
        _finite(result["frame_rate_hz"], "camera frame rate", 0.0)
    for field in ("resolution", "vial_roi"):
        if result[field] != "unavailable" and (
            not isinstance(result[field], list)
            or len(result[field]) not in {2, 4}
            or any(type(item) is not int or item < 0 for item in result[field])
        ):
            raise ContractError(f"camera {field} is invalid")
    if production and any(result[key] == "unavailable" for key in ("resolution", "frame_rate_hz", "viewpoint_id", "vial_roi")):
        raise ContractError("production camera metadata is incomplete")
    return result


def build_vial1_bundle(
    cohort: object, image_locators: object, *, tracker_receipts: object,
    production: bool = False,
) -> BuiltBundle:
    """Build a path-free bundle and verify a separate internal locator map."""
    if not isinstance(cohort, Mapping) or not isinstance(image_locators, Mapping):
        raise ContractError("cohort and locators must be objects")
    _scan(cohort)
    cohort_fields = {
        "schema_version", "cohort_id", "partition", "recipe_id",
        "previous_recipe_id", "camera_metadata", "stages", "vials",
        "deterministic_decision_receipt",
    }
    if set(cohort) not in (cohort_fields, cohort_fields | {"target_vial_position"}):
        raise ContractError("cohort fields are not closed")
    if cohort.get("schema_version") != "wp7-vlm-cohort-v1":
        raise ContractError("cohort schema is invalid")
    if cohort.get("target_vial_position", 1) != 1:
        raise ContractError("only vial 1 may be scored")
    vials, stages = cohort.get("vials"), cohort.get("stages")
    if not isinstance(vials, list) or len(vials) != 3 or {v.get("vial_position") for v in vials if isinstance(v, Mapping)} != {1, 2, 3}:
        raise ContractError("complete three vials are required")
    if not isinstance(stages, list) or not stages:
        raise ContractError("cohort stages are required")
    stage_ids = [stage.get("stage_id") for stage in stages if isinstance(stage, Mapping)]
    tracker = _tracker_cohort(tracker_receipts, cohort.get("cohort_id"), vials)
    camera = _camera(cohort.get("camera_metadata", {}), production)
    if camera["resolution"] == "unavailable" or camera["vial_roi"] == "unavailable":
        raise ContractError("camera resolution and frozen vial ROI are required")
    images, evidence, bound = [], [], {}
    for vial, tracker_receipt in zip(
        sorted(vials, key=lambda row: row["vial_position"]), tracker, strict=True
    ):
        if set(vial) != {"vial_position", "vial_id", "stages"}:
            raise ContractError("vial fields are not closed")
        vial_stages = vial.get("stages")
        if (
            not isinstance(vial_stages, list)
            or [row.get("stage_id") for row in vial_stages if isinstance(row, Mapping)] != stage_ids
            or len(tracker_receipt["stages"]) != len(stage_ids)
        ):
            raise ContractError("vial stage/cohort mismatch")
        for stage_index, (stage, tracker_stage) in enumerate(
            zip(vial_stages, tracker_receipt["stages"], strict=True)
        ):
            if tracker_stage["fill_step"] != stage_index + 1:
                raise ContractError("tracker fill-step order is invalid")
            for phase in PHASES:
                image_id = tracker_stage["images"][phase]
                locator = image_locators.get(image_id)
                if not isinstance(locator, Mapping):
                    raise ContractError(f"unbound image ID: {image_id}")
                locator_fields = {
                    "path", "sha256", "source_full_frame_sha256", "crop_pixel_sha256",
                    "crop_box_xyxy", "source_native_dimensions", "crop_native_dimensions",
                }
                if set(locator) != locator_fields:
                    raise ContractError("image locator fields are invalid")
                path, digest = Path(locator["path"]), locator["sha256"]
                if path.is_symlink() or not path.is_file() or not isinstance(digest, str) or _SHA.fullmatch(digest) is None or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ContractError("image locator/hash binding failed")
                if (
                    any(_SHA.fullmatch(str(locator[field])) is None for field in (
                        "source_full_frame_sha256", "crop_pixel_sha256"
                    ))
                    or locator["crop_box_xyxy"] != camera["vial_roi"]
                    or locator["source_native_dimensions"] != camera["resolution"]
                    or locator["crop_native_dimensions"] != [
                        camera["vial_roi"][2] - camera["vial_roi"][0],
                        camera["vial_roi"][3] - camera["vial_roi"][1],
                    ]
                ):
                    raise ContractError("image crop/native-frame binding differs from frozen ROI")
                if image_id in bound:
                    raise ContractError("tracker image ID is reused across the cohort")
                bound[image_id] = BoundImageLocator(
                    path.resolve(), digest, locator["source_full_frame_sha256"],
                    locator["crop_pixel_sha256"], tuple(locator["crop_box_xyxy"]),
                    tuple(locator["source_native_dimensions"]),
                    tuple(locator["crop_native_dimensions"]),
                )
                images.append({
                    "image_id": image_id, "vial_position": vial["vial_position"],
                    "stage_id": stage["stage_id"], "phase": phase,
                    "source_full_frame_sha256": locator["source_full_frame_sha256"],
                    "crop_sha256": digest,
                    "crop_pixel_sha256": locator["crop_pixel_sha256"],
                    "crop_box_xyxy": locator["crop_box_xyxy"],
                    "source_native_dimensions": locator["source_native_dimensions"],
                    "crop_native_dimensions": locator["crop_native_dimensions"],
                })
            if not isinstance(stage.get("measurement"), Mapping):
                raise ContractError("deterministic measurement is missing")
            measurement_base_fields = {
                "evidence_id", "median_lab", "mad_lab", "iqr_lab", "limit_codes",
            }
            measurement_fields = measurement_base_fields | {
                "delta_e76_vial2", "delta_e76_vial3",
                "deterministic_disposition",
            }
            allowed_fields = (
                {frozenset(measurement_fields)} if vial["vial_position"] == 1
                else {frozenset(measurement_base_fields), frozenset(measurement_fields)}
            )
            if frozenset(stage["measurement"]) not in allowed_fields:
                raise ContractError("deterministic measurement fields are not closed")
            projected_measurement = _measurement(
                stage["measurement"], target=vial["vial_position"] == 1
            )
            evidence.append({
                "vial_position": vial["vial_position"],
                "evidence_role": "target" if vial["vial_position"] == 1 else "reference",
                "stage_id": stage["stage_id"],
                **projected_measurement,
            })
    target_rows = [row for row in evidence if row["vial_position"] == 1]
    reference_rows = {
        position: [row for row in evidence if row["vial_position"] == position]
        for position in (2, 3)
    }
    for index, target in enumerate(target_rows):
        for position, field in ((2, "delta_e76_vial2"), (3, "delta_e76_vial3")):
            computed = _delta_e76(
                target["median_lab"], reference_rows[position][index]["median_lab"]
            )
            if not math.isclose(float(target[field]), computed, rel_tol=0.0, abs_tol=1e-9):
                raise ContractError(f"target {field} does not match same-stage median Lab deltaE76")
            target[field] = computed
    deterministic = _validate_deterministic_receipt(
        cohort["deterministic_decision_receipt"],
        cohort_id=cohort.get("cohort_id"),
        target_vial_id=next(row["vial_id"] for row in vials if row["vial_position"] == 1),
        target_rows=target_rows,
    )
    allowed = {"stage_id", "stage_ordinal", "tank_id", "programmed_ml", "dispense_order", "stage_timing_seconds", "stream"}
    safe_stages = []
    target_tracker_stages = tracker[0]["stages"]
    for stage, tracker_stage in zip(stages, target_tracker_stages, strict=True):
        if set(stage) != allowed:
            raise ContractError("stage fields are invalid or forbidden")
        timing, stream = stage["stage_timing_seconds"], stage["stream"]
        if not isinstance(timing, list) or len(timing) != 2 or _finite(timing[1], "stage timing") <= _finite(timing[0], "stage timing") or not isinstance(stream, Mapping) or set(stream) != {"evidence_id", "median_lab", "spread_lab"}:
            raise ContractError("stage timing/stream is invalid")
        programmed_ml = _finite(stage["programmed_ml"], "programmed mL", 0.0)
        if programmed_ml <= 0 or stage["tank_id"].lower() != tracker_stage["tank_id"] or programmed_ml != tracker_stage["expected_dispense_ml"]:
            raise ContractError("tracker and recipe stage differ")
        safe_stages.append({
            **stage,
            "programmed_ml": programmed_ml,
            "tank_cluster_consistent": tracker_stage["tank_cluster_consistent"],
            "stream": {
                **stream,
                "median_lab": _lab(tracker_stage["stream_lab"], "stream Lab"),
                "spread_lab": _lab(tracker_stage["stream_spread_lab"], "stream spread"),
            },
            "camera_fill_state": {
                key: tracker_stage[key]
                for key in (
                    "fill_step", "tank_id", "tank_number", "stream_step_index",
                    "expected_dispense_ml", "expected_cumulative_ml",
                    "twin_geometry_equivalent_volume_ml", "measured_volume_ml",
                    "agreement",
                )
            },
        })
    payload = {
        "schema_version": "wp7-vlm-prompt-bundle-v2", "cohort_id": cohort.get("cohort_id"),
        "partition": cohort.get("partition"), "recipe_id": cohort.get("recipe_id"), "previous_recipe_id": cohort.get("previous_recipe_id"),
        "previous_recipe_direction_support": dict(
            _NO_PREVIOUS_RECIPE_DIRECTION_SUPPORT
        ),
        "target_vial_position": 1, "camera_metadata": camera,
        "stages": safe_stages, "images": images, "deterministic_evidence": evidence,
        "tracker_binding": {
            "receipt_sha256": [item["receipt_sha256"] for item in tracker],
            "source_artifact_sha256": [item["source_artifact_sha256"] for item in tracker],
            "calibration_sha256": tracker[0]["calibration_sha256"],
            "claim_boundary": tracker[0]["claim_boundary"],
            "validator_source_sha256": hashlib.sha256(
                Path(__file__).with_name("wp7_fill_step_tracker.py").read_bytes()
            ).hexdigest(),
        },
        "deterministic_binding": deterministic,
    }
    _scan(payload)
    encoded_payload = _canonical(payload).decode()
    user = USER_PROMPT_TEMPLATE.format(schema=json.dumps(RESPONSE_SCHEMA, sort_keys=True), bundle=encoded_payload)
    return BuiltBundle(payload, bound, SYSTEM_PROMPT, user)


def _strip_one_bounded_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", text, re.I)
    if match:
        return match.group(1)
    if "```" in text:
        raise ContractError("malformed response fence")
    return text


def _has_unsupported_previous_recipe_direction_claim(
    text: str, previous_recipe_id: object,
) -> bool:
    """Detect a positive directional assertion, not a support disclaimer."""
    for sentence in re.split(r"[.!?;\n]+", text):
        names_previous = _PREVIOUS_RECIPE_TEXT.search(sentence) is not None
        names_identifier = (
            isinstance(previous_recipe_id, str)
            and bool(previous_recipe_id)
            and previous_recipe_id.casefold() in sentence.casefold()
        )
        if (
            (names_previous or names_identifier)
            and _DIRECTIONAL_CLAIM_TEXT.search(sentence) is not None
        ):
            return True
    return False


def parse_model_response(
    raw: object, *, repository: str | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, str) or len(raw.encode()) > 65536:
        raise ContractError("model response is not bounded text")
    text = raw.strip()
    mode = MODEL_REASONING_MODES.get(repository) if repository is not None else None
    if repository is not None and mode is None:
        raise ContractError("model response repository is not pinned")
    if mode == "native_think_answer":
        if text.count("<answer>") != 1 or text.count("</answer>") != 1:
            raise ContractError("native reasoning requires exactly one answer element")
        if text.count("<think>") != 1 or text.count("</think>") != 1:
            raise ContractError("native reasoning requires one think before answer")
        match = re.fullmatch(
            r"<think>([\s\S]*?)</think><answer>([\s\S]*?)</answer>", text
        )
        if match is None:
            raise ContractError("native reasoning think/answer order is invalid")
        text = _strip_one_bounded_fence(match.group(2).strip())
    elif mode == "schema_json":
        if "```" in text or "<answer>" in text or "<think>" in text:
            raise ContractError("Qwen response must be strict JSON without wrappers")
    else:
        # Compatibility for callers without a model identity; runtime always supplies one.
        text = _strip_one_bounded_fence(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError("model response is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ContractError("model response must be one object")
    return value


def guard_model_response(value: object, bundle: object) -> dict[str, Any]:
    base = {"vote": None, "discard_score": None, "guard_status": "excluded", "guard_reasons": []}
    if not isinstance(value, Mapping) or not isinstance(bundle, Mapping):
        return {**base, "guard_reasons": ["response_or_bundle_type_invalid"]}
    reasons = []
    if tuple(value) != RESPONSE_FIELD_ORDER:
        reasons.append("response_fields_or_order_invalid")
    observations = value.get("observations")
    comparisons = value.get("per_stage_comparison")
    allowed_evidence = {
        row.get("evidence_id") for row in bundle.get("deterministic_evidence", [])
        if isinstance(row, Mapping)
    }
    allowed_images = {
        row.get("image_id") for row in bundle.get("images", [])
        if isinstance(row, Mapping)
    }
    allowed_ids = allowed_evidence | allowed_images
    if not isinstance(observations, list) or not observations or any(
        not isinstance(item, Mapping) or tuple(item) != ("text", "bundle_ids")
        or not isinstance(item["text"], str) or not item["text"]
        or not isinstance(item["bundle_ids"], list) or not item["bundle_ids"]
        for item in observations
    ):
        reasons.append("observations_invalid")
    elif any(
        any(
            not isinstance(identifier, str) or identifier not in allowed_ids
            for identifier in item["bundle_ids"]
        )
        for item in observations
    ):
        reasons.append("observation_bundle_id_missing")
    else:
        try:
            for item in observations:
                _scan(item["text"])
        except ContractError:
            reasons.append("observation_path_or_forbidden_text")
    comparison_fields = (
        "stage_id", "phase_and_fill_match", "vial1_vs_vial2", "vial1_vs_vial3",
        "deterministic_stage_disposition", "stream_tank_consistency",
        "previous_recipe_direction", "reliability", "bundle_ids",
    )
    physical_text_fields = ("phase_and_fill_match",)
    supported_stage_ids = [
        row.get("stage_id") for row in bundle.get("stages", [])
        if isinstance(row, Mapping)
    ]
    direction_support = bundle.get("previous_recipe_direction_support")
    if direction_support != _NO_PREVIOUS_RECIPE_DIRECTION_SUPPORT:
        reasons.append("previous_recipe_direction_support_invalid")
    if not isinstance(comparisons, list) or any(
        not isinstance(row, Mapping) or tuple(row) != comparison_fields
        for row in comparisons
    ):
        reasons.append("per_stage_comparison_invalid")
    elif (
        len(comparisons) != len(supported_stage_ids)
        or [row["stage_id"] for row in comparisons] != supported_stage_ids
        or len(set(supported_stage_ids)) != len(supported_stage_ids)
    ):
        reasons.append("per_stage_comparison_coverage_invalid")
    else:
        for row in comparisons:
            stage_id = row["stage_id"]
            if any(
                not isinstance(row[field], str) or not row[field]
                for field in physical_text_fields
            ):
                reasons.append("per_stage_comparison_text_invalid")
                break
            if any(row[field] not in REFERENCE_RELATIONS for field in (
                "vial1_vs_vial2", "vial1_vs_vial3",
            )):
                reasons.append("per_stage_reference_relation_invalid")
                break
            if row["stream_tank_consistency"] not in STREAM_TANK_CONSISTENCY:
                reasons.append("stream_tank_consistency_invalid")
                break
            if row["previous_recipe_direction"] not in PREVIOUS_RECIPE_DIRECTIONS:
                reasons.append("previous_recipe_direction_invalid")
                break
            if (
                not isinstance(row["deterministic_stage_disposition"], str)
                or row["deterministic_stage_disposition"] not in DISPOSITIONS
            ):
                reasons.append("deterministic_stage_disposition_invalid")
                break
            if row["reliability"] not in STAGE_RELIABILITY:
                reasons.append("per_stage_comparison_reliability_invalid")
                break
            try:
                for field in physical_text_fields:
                    _scan(row[field])
            except ContractError:
                reasons.append("per_stage_comparison_path_or_forbidden_text")
                break
            target_rows = [
                evidence for evidence in bundle.get("deterministic_evidence", [])
                if isinstance(evidence, Mapping)
                and evidence.get("stage_id") == stage_id
                and evidence.get("vial_position") == 1
            ]
            prompt_stages = [
                stage for stage in bundle.get("stages", [])
                if isinstance(stage, Mapping) and stage.get("stage_id") == stage_id
            ]
            if (
                len(prompt_stages) != 1
                or type(prompt_stages[0].get("tank_cluster_consistent")) is not bool
                or row["stream_tank_consistency"] != (
                    "consistent"
                    if prompt_stages[0].get("tank_cluster_consistent") is True
                    else "inconsistent"
                )
            ):
                reasons.append("stream_tank_consistency_mismatch")
            if row["previous_recipe_direction"] != "not_assessable":
                reasons.append("previous_recipe_direction_unsupported")
            try:
                expected_relations = (
                    _reference_relation(target_rows[0]["delta_e76_vial2"]),
                    _reference_relation(target_rows[0]["delta_e76_vial3"]),
                )
            except (ContractError, IndexError, KeyError):
                reasons.append("per_stage_reference_measurement_invalid")
                break
            if (
                row["vial1_vs_vial2"], row["vial1_vs_vial3"]
            ) != expected_relations:
                reasons.append("per_stage_reference_relation_mismatch")
            if (
                len(target_rows) != 1
                or row["deterministic_stage_disposition"]
                != target_rows[0].get("deterministic_disposition")
            ):
                reasons.append("deterministic_stage_disposition_mismatch")
            stage_ids = {
                evidence.get("evidence_id")
                for evidence in bundle.get("deterministic_evidence", [])
                if isinstance(evidence, Mapping) and evidence.get("stage_id") == stage_id
            } | {
                image.get("image_id")
                for image in bundle.get("images", [])
                if isinstance(image, Mapping) and image.get("stage_id") == stage_id
            }
            cited = row["bundle_ids"]
            if (
                not isinstance(cited, list) or not cited
                or any(not isinstance(item, str) or item not in stage_ids for item in cited)
            ):
                reasons.append("per_stage_comparison_bundle_ids_invalid")
                break
    disposition, confidence = value.get("disposition"), value.get("disposition_confidence")
    if disposition not in DISPOSITIONS: reasons.append("disposition_invalid")
    try:
        confidence = _finite(confidence, "disposition confidence", 0.0)
        if confidence > 1: reasons.append("disposition_confidence_out_of_range")
        if confidence < 0.6 and disposition != "insufficient_evidence":
            reasons.append("below_0_6_requires_insufficient_evidence")
        if (
            confidence > 0.8 and disposition != "insufficient_evidence"
            and isinstance(comparisons, list) and len(comparisons) < 2
        ):
            reasons.append("single_stage_confidence_above_0_8")
        if confidence > 0.8 and isinstance(comparisons, list) and any(
            not isinstance(row, Mapping)
            or row.get("reliability") != "clear_consistent"
            for row in comparisons
        ):
            reasons.append("non_clear_stage_confidence_above_0_8")
    except ContractError: reasons.append("disposition_confidence_invalid")
    explanation = value.get("explanation")
    if not isinstance(explanation, str) or not explanation: reasons.append("explanation_invalid")
    else:
        try: _scan(explanation)
        except ContractError: reasons.append("path_or_forbidden_text")
    if isinstance(explanation, str):
        mentioned = set(re.findall(r"(?:IMG|EV)-[A-Za-z0-9_-]+", explanation))
        if not mentioned.issubset(allowed_evidence | allowed_images):
            reasons.append("explanation_citation_invalid")
    prose = [explanation] if isinstance(explanation, str) else []
    if isinstance(observations, list):
        prose.extend(
            item.get("text") for item in observations
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        )
    previous_recipe_id = bundle.get("previous_recipe_id")
    if any(
        _has_unsupported_previous_recipe_direction_claim(item, previous_recipe_id)
        for item in prose
    ):
        reasons.append("unsupported_previous_recipe_direction_text")
    for field, allowed_ids in (("evidence_citations", allowed_evidence), ("image_citations", allowed_images)):
        citations = value.get(field)
        if not isinstance(citations, list) or not citations or any(not isinstance(item, str) or item not in allowed_ids for item in citations): reasons.append(f"{field}_invalid")
    result = {**dict(value), **base, "guard_reasons": list(dict.fromkeys(reasons))}
    if not reasons:
        result["guard_status"] = "admitted"
        discard_score = (
            None if disposition == "insufficient_evidence"
            else confidence if disposition == "discard" else 1.0 - confidence
        )
        result["discard_score"] = discard_score
        result["vote"] = discard_score
    return result


def aggregate_decision(bundle: object, model_attempts: object) -> dict[str, Any]:
    if (
        not isinstance(bundle, Mapping)
        or not isinstance(bundle.get("deterministic_binding"), Mapping)
        or bundle["deterministic_binding"].get("disposition") not in DISPOSITIONS
        or not isinstance(model_attempts, Sequence)
        or isinstance(model_attempts, (str, bytes))
        or any(not isinstance(row, Mapping) for row in model_attempts)
    ):
        raise ContractError("aggregation inputs are invalid")
    deterministic_raw = bundle["deterministic_binding"]
    target_rows = [
        row for row in bundle.get("deterministic_evidence", [])
        if isinstance(row, Mapping) and row.get("vial_position") == 1
    ]
    deterministic = _validate_deterministic_receipt(
        deterministic_raw, cohort_id=bundle.get("cohort_id"),
        target_vial_id=deterministic_raw.get("target_vial_id"),
        target_rows=target_rows,
    )
    det_vote = {"keep": 0.0, "discard": 1.0, "insufficient_evidence": None}[deterministic["disposition"]]
    sources = [{
        "source_id": "deterministic", "receipt_sha256": deterministic["receipt_sha256"],
        "disposition": deterministic["disposition"], "limit_codes": deterministic["limit_codes"],
        "discard_score": det_vote, "vote": det_vote,
        "guard_status": "admitted" if det_vote is not None else "no_vote",
    }]
    pinned = {row["repository"]: row["revision"] for row in MODELS}
    seen: set[str] = set()
    for attempt in model_attempts:
        if set(attempt) != {"repository", "revision", "raw_response"}:
            raise ContractError("model attempt fields are not closed")
        repository, revision = attempt["repository"], attempt["revision"]
        if repository not in pinned or pinned[repository] != revision:
            raise ContractError("model attempt is not an exact pinned model")
        if repository in seen:
            raise ContractError("duplicate model source")
        seen.add(repository)
        try:
            parsed = parse_model_response(
                attempt["raw_response"], repository=str(repository)
            )
            guarded = guard_model_response(parsed, bundle)
        except ContractError as exc:
            guarded = {
                "vote": None, "discard_score": None, "guard_status": "excluded",
                "guard_reasons": [f"parse_failed:{exc}"],
            }
        sources.append({
            "source_id": repository, "repository": repository, "revision": revision,
            **guarded,
        })
    votes = [
        float(row["vote"])
        for row in sources
        if row.get("guard_status") == "admitted"
        and isinstance(row.get("vote"), (int, float))
        and not isinstance(row.get("vote"), bool)
        and math.isfinite(float(row["vote"]))
        and 0 <= float(row["vote"]) <= 1
    ]
    mean = sum(votes) / len(votes) if votes else None
    final = "insufficient_evidence" if len(votes) < 2 else ("discard" if mean >= 0.5 else "keep")
    return {"schema_version": "wp7-vlm-aggregation-receipt-v1", "aggregation": dict(AGGREGATION), "sources": sources, "valid_source_count": len(votes), "mean_discard_score": mean, "final_disposition": final}


def run_sequential_vlm_runtime(
    built: BuiltBundle, model_snapshots: Mapping[str, str | Path], *,
    runtime_loader: Any = None, runtime_invoker: Any = None,
    constrained_decoder_builder: Any = None,
) -> dict[str, Any]:
    """Deliver every ordered image to each exact pinned model, one model at a time."""
    from fluidmix.model_bakeoff_runner import (
        ImageInput, invoke_stock_runtime, load_stock_runtime,
    )

    if not isinstance(built, BuiltBundle) or set(model_snapshots) != {
        row["repository"] for row in MODELS
    }:
        raise ContractError("runtime requires snapshots for exactly the pinned models")
    runtime_loader = runtime_loader or load_stock_runtime
    runtime_invoker = runtime_invoker or invoke_stock_runtime
    if constrained_decoder_builder is None:
        from fluidmix.wp7_vlm_constrained import build_logits_processor
        constrained_decoder_builder = build_logits_processor
    images = tuple(
        ImageInput(
            path=built.image_locators[row["image_id"]].path,
            sha256=built.image_locators[row["image_id"]].sha256,
            image_id=row["image_id"],
        )
        for row in built.prompt_payload["images"]
    )
    prompt = f"{built.system_prompt}\n\n{built.user_prompt}"
    attempts = []
    for spec in MODELS:
        repository, revision = spec["repository"], spec["revision"]
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = GENERATION_CONFIG[
            "cublas_workspace_config"
        ]
        runtime = runtime_loader(repository, revision, model_snapshots[repository])
        try:
            _configure_runtime_determinism(runtime)
            logits_processor, _decoder_receipt = constrained_decoder_builder(
                runtime, repository=repository, response_schema=RESPONSE_SCHEMA
            )
            row = runtime_invoker(
                runtime, prompt, images,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                logits_processor=logits_processor,
                processor_pixels=GENERATION_CONFIG["processor_pixels"],
            )
            raw = row.get("response_raw") if isinstance(row, Mapping) and row.get("status") == "completed" else None
            attempts.append({"repository": repository, "revision": revision, "raw_response": raw})
        finally:
            del runtime
    return aggregate_decision(built.prompt_payload, attempts)


def _bind_materialization_receipt(
    path: str | Path, *, development_records: Sequence[Mapping[str, object]],
    trackers: Sequence[Mapping[str, object]],
) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ContractError("development materialization receipt is absent")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("development materialization receipt is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise ContractError("development materialization receipt is invalid")
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    if (
        value.get("schema_version")
        != "wp7-vlm-development-bundle-materialization-v1"
        or value.get("status") != "complete"
        or claimed != _digest(body)
    ):
        raise ContractError("development materialization receipt identity is invalid")
    expected_manifests = {
        str(row["dataset_version"]): str(row["manifest_sha256"])
        for row in development_records
    }
    source_bindings = value.get("source_bindings")
    if (
        len(expected_manifests) != len(development_records)
        or value.get("development_manifest_sha256") != expected_manifests
        or not isinstance(source_bindings, Mapping)
        or source_bindings.get("deterministic_config_sha256")
        != DETERMINISTIC_CONFIG_SHA256
        or source_bindings.get("tracker_receipt_sha256")
        != [tracker["receipt_sha256"] for tracker in trackers]
        or source_bindings.get("tracker_source_sha256")
        != hashlib.sha256(
            Path(__file__).with_name("wp7_fill_step_tracker.py").read_bytes()
        ).hexdigest()
    ):
        raise ContractError("development materialization receipt bindings differ")
    return hashlib.sha256(raw).hexdigest()


def create_development_freeze(
    path: str | Path, development_inputs: Sequence[str | Path], *,
    tracker_receipts: Sequence[Mapping[str, object]] | None = None,
    materialization_receipt: str | Path,
) -> dict[str, Any]:
    """Publish a freeze only with accepted development tracker receipts."""
    if tracker_receipts is None:
        raise ContractError("freeze requires three accepted tracker receipts")
    trackers = _validate_tracker_cohort_receipts(tracker_receipts)
    records = []
    for raw in development_inputs:
        item = Path(raw)
        if item.is_symlink() or not item.is_file():
            raise ContractError("development manifest is absent")
        raw_bytes = item.read_bytes()
        try:
            manifest = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise ContractError("development manifest is invalid JSON") from exc
        fields = {"schema_version", "dataset_version", "partition", "data_origin", "artifacts"}
        if not isinstance(manifest, Mapping) or set(manifest) != fields:
            raise ContractError("development manifest fields are not closed")
        if (
            manifest.get("schema_version") != "wp7-vlm-development-input-manifest-v1"
            or manifest.get("dataset_version") not in {"V3", "V4"}
            or manifest.get("partition") not in {"twin_train", "twin_validation"}
            or manifest.get("data_origin") != "synthetic_twin"
        ):
            raise ContractError("development manifest must be V3/V4 synthetic twin train/validation")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ContractError("development manifest has no artifacts")
        safe_artifacts = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping) or set(artifact) != {"artifact_id", "path", "sha256"}:
                raise ContractError("development manifest artifact fields are not closed")
            artifact_id = artifact["artifact_id"]
            relative = artifact["path"]
            expected = artifact["sha256"]
            if (
                not isinstance(artifact_id, str) or not artifact_id
                or not isinstance(relative, str) or not relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not isinstance(expected, str) or _SHA.fullmatch(expected) is None
            ):
                raise ContractError("development manifest artifact binding is invalid")
            lowered = _canonical(artifact).decode().lower()
            if any(token in lowered for token in ("video5", "video_5", "video-5", "evaluation", "real_data", "test_data")):
                raise ContractError("development manifest references forbidden held-out data")
            bound = item.parent / relative
            if bound.is_symlink() or not bound.is_file() or hashlib.sha256(bound.read_bytes()).hexdigest() != expected:
                raise ContractError("development manifest artifact hash differs")
            safe_artifacts.append({"artifact_id": artifact_id, "sha256": expected})
        records.append({
            "manifest_name": item.name,
            "manifest_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "dataset_version": manifest["dataset_version"],
            "partition": manifest["partition"],
            "data_origin": manifest["data_origin"],
            "artifacts": safe_artifacts,
        })
    if {item["dataset_version"] for item in records} != {"V3", "V4"}:
        raise ContractError("development manifests must cover both V3 and V4")
    if len({item["manifest_name"] for item in records}) != len(records):
        raise ContractError("development manifest names must be unique")
    materialization_receipt_sha256 = _bind_materialization_receipt(
        materialization_receipt,
        development_records=records,
        trackers=trackers,
    )
    output = Path(path)
    if output.exists() or output.is_symlink(): raise ContractError("refusing to overwrite freeze")
    freeze = {
        "schema_version": "wp7-vlm-development-freeze-v8", "development_only": True,
        "decision_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "deterministic_policy": {
            "config_sha256": DETERMINISTIC_CONFIG_SHA256,
            **DETERMINISTIC_POLICY,
        },
        "generation_config": dict(GENERATION_CONFIG),
        "development_materialization_receipt_sha256": materialization_receipt_sha256,
        "tracker_schema_version": "wp7-camera-fill-step-receipt-v1",
        "tracker_cohort_id": trackers[0]["cohort_id"],
        "tracker_vial_ids": [item["vial_id"] for item in trackers],
        "tracker_receipt_sha256": [item["receipt_sha256"] for item in trackers],
        "tracker_calibration_sha256": sorted({item["calibration_sha256"] for item in trackers}),
        "tracker_recipe_artifact_sha256": sorted({item["recipe_artifact_sha256"] for item in trackers}),
        "tracker_claim_boundary": trackers[0]["claim_boundary"],
        "tracker_recipe_stages": [
            {key: stage[key] for key in _TRACKER_RECIPE_KEYS}
            for stage in trackers[0]["stages"]
        ],
        "tracker_source_artifact_sha256": sorted({item["source_artifact_sha256"] for item in trackers}),
        "tracker_validator_source_sha256": hashlib.sha256(
            Path(__file__).with_name("wp7_fill_step_tracker.py").read_bytes()
        ).hexdigest(),
        "prompt_id": "wp7-grounded-sequential-mixture-v5",
        "prompt_sha256": _digest({"system": SYSTEM_PROMPT, "user_template": USER_PROMPT_TEMPLATE}),
        "response_schema_id": "wp7-vlm-reasoned-response-v4",
        "response_schema_sha256": _digest(RESPONSE_SCHEMA),
        "parser_id": "model-specific-json-or-one-think-answer-v1",
        "reasoning_modes": dict(MODEL_REASONING_MODES),
        "constrained_decoding": {
            "backend": "xgrammar", "version": "0.2.8", "license": "Apache-2.0",
            "decoder_id": "xgrammar-0.2.8-wp7-structural-v1",
        },
        "aggregation": dict(AGGREGATION), "models": list(MODELS),
        "development_inputs": sorted(records, key=lambda row: row["manifest_name"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as stream: stream.write(json.dumps(freeze, sort_keys=True, indent=2) + "\n")
    return freeze


def _validate_freeze(value: object) -> None:
    if not isinstance(value, Mapping): raise ContractError("freeze is invalid")
    fields = {
        "schema_version", "development_only", "decision_source_sha256",
        "deterministic_policy", "generation_config",
        "development_materialization_receipt_sha256",
        "tracker_schema_version", "tracker_cohort_id", "tracker_vial_ids",
        "tracker_receipt_sha256", "tracker_calibration_sha256",
        "tracker_recipe_artifact_sha256", "tracker_claim_boundary",
        "tracker_recipe_stages", "tracker_source_artifact_sha256",
        "tracker_validator_source_sha256",
        "prompt_id", "prompt_sha256", "response_schema_id",
        "response_schema_sha256", "parser_id", "reasoning_modes",
        "constrained_decoding", "aggregation",
        "models", "development_inputs",
    }
    if set(value) != fields:
        raise ContractError("freeze fields are not closed")
    expected = {
        "schema_version": "wp7-vlm-development-freeze-v8", "development_only": True,
        "deterministic_policy": {
            "config_sha256": DETERMINISTIC_CONFIG_SHA256,
            **DETERMINISTIC_POLICY,
        },
        "generation_config": GENERATION_CONFIG,
        "tracker_schema_version": "wp7-camera-fill-step-receipt-v1",
        "tracker_claim_boundary": TRACKER_CLAIM_BOUNDARY,
        "prompt_id": "wp7-grounded-sequential-mixture-v5",
        "prompt_sha256": _digest({"system": SYSTEM_PROMPT, "user_template": USER_PROMPT_TEMPLATE}),
        "response_schema_id": "wp7-vlm-reasoned-response-v4",
        "response_schema_sha256": _digest(RESPONSE_SCHEMA),
        "parser_id": "model-specific-json-or-one-think-answer-v1",
        "reasoning_modes": MODEL_REASONING_MODES,
        "constrained_decoding": {
            "backend": "xgrammar", "version": "0.2.8", "license": "Apache-2.0",
            "decoder_id": "xgrammar-0.2.8-wp7-structural-v1",
        },
        "aggregation": AGGREGATION, "models": list(MODELS),
    }
    if any(value.get(key) != item for key, item in expected.items()): raise ContractError("freeze constants do not match code")
    if _SHA.fullmatch(str(value.get("development_materialization_receipt_sha256"))) is None:
        raise ContractError("freeze materialization receipt hash is invalid")
    if value.get("decision_source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ContractError("freeze source does not match code")
    if not all(isinstance(value.get(field), list) and value[field] for field in (
        "tracker_vial_ids", "tracker_receipt_sha256", "tracker_calibration_sha256",
        "tracker_recipe_artifact_sha256", "tracker_source_artifact_sha256",
    )):
        raise ContractError("freeze tracker binding is incomplete")
    if (
        not _opaque_identifier(value.get("tracker_cohort_id"))
        or len(value["tracker_vial_ids"]) != 3
        or any(not _opaque_identifier(item) for item in value["tracker_vial_ids"])
        or len(set(value["tracker_vial_ids"])) != 3
    ):
        raise ContractError("freeze tracker identities are invalid")
    hash_fields = (
        "tracker_receipt_sha256", "tracker_calibration_sha256",
        "tracker_recipe_artifact_sha256", "tracker_source_artifact_sha256",
    )
    if any(
        not isinstance(item, str) or _SHA.fullmatch(item) is None
        for field in hash_fields
        for item in value[field]
    ):
        raise ContractError("freeze tracker hashes are invalid")
    if (
        len(value["tracker_receipt_sha256"]) != 3
        or len(set(value["tracker_receipt_sha256"])) != 3
        or len(value["tracker_calibration_sha256"]) != 1
        or len(value["tracker_recipe_artifact_sha256"]) != 1
        or len(value["tracker_source_artifact_sha256"]) != 3
        or len(set(value["tracker_source_artifact_sha256"])) != 3
        or value["tracker_source_artifact_sha256"] != sorted(value["tracker_source_artifact_sha256"])
    ):
        raise ContractError("freeze tracker hashes are invalid")
    recipe_stages = value.get("tracker_recipe_stages")
    if not isinstance(recipe_stages, list) or not recipe_stages:
        raise ContractError("freeze tracker recipe stages are invalid")
    previous_cumulative = 0.0
    for ordinal, stage in enumerate(recipe_stages, start=1):
        if not isinstance(stage, Mapping) or set(stage) != set(_TRACKER_RECIPE_KEYS):
            raise ContractError("freeze tracker recipe stage fields are not closed")
        if (
            stage["fill_step"] != ordinal
            or stage["stream_step_index"] != ordinal
            or not isinstance(stage["tank_id"], str)
            or not stage["tank_id"]
            or isinstance(stage["tank_number"], bool)
            or not isinstance(stage["tank_number"], int)
        ):
            raise ContractError("freeze tracker recipe stages are invalid")
        dispense = _finite(stage["expected_dispense_ml"], "expected dispense", 0.0)
        cumulative = _finite(stage["expected_cumulative_ml"], "expected cumulative", 0.0)
        if dispense <= 0.0 or cumulative != previous_cumulative + dispense:
            raise ContractError("freeze tracker recipe stages are invalid")
        previous_cumulative = cumulative
    tracker_source = Path(__file__).with_name("wp7_fill_step_tracker.py")
    if value["tracker_validator_source_sha256"] != hashlib.sha256(tracker_source.read_bytes()).hexdigest():
        raise ContractError("freeze tracker validator source differs")
    if not isinstance(value["development_inputs"], list) or not value["development_inputs"]:
        raise ContractError("freeze development inputs are absent")
    if {item.get("dataset_version") for item in value["development_inputs"] if isinstance(item, Mapping)} != {"V3", "V4"}:
        raise ContractError("freeze development inputs do not cover V3/V4")
    for item in value["development_inputs"]:
        fields = {"manifest_name", "manifest_sha256", "dataset_version", "partition", "data_origin", "artifacts"}
        if (
            not isinstance(item, Mapping) or set(item) != fields
            or not isinstance(item["manifest_name"], str)
            or _SHA.fullmatch(str(item["manifest_sha256"])) is None
            or item["dataset_version"] not in {"V3", "V4"}
            or item["partition"] not in {"twin_train", "twin_validation"}
            or item["data_origin"] != "synthetic_twin"
            or not isinstance(item["artifacts"], list) or not item["artifacts"]
            or any(
                not isinstance(artifact, Mapping)
                or set(artifact) != {"artifact_id", "sha256"}
                or not isinstance(artifact["artifact_id"], str)
                or _SHA.fullmatch(str(artifact["sha256"])) is None
                for artifact in item["artifacts"]
            )
        ):
            raise ContractError("freeze development input binding is invalid")


def load_production_bundle(
    bundle_path: str | Path, freeze_path: str | Path, *, expected_freeze_sha256: str
) -> dict[str, Any]:
    """Validate freeze before touching the production bundle path."""
    freeze_file = Path(freeze_path)
    if freeze_file.is_symlink() or not freeze_file.is_file(): raise ContractError("matching development freeze is required")
    raw_freeze = freeze_file.read_bytes()
    if _SHA.fullmatch(expected_freeze_sha256) is None or hashlib.sha256(raw_freeze).hexdigest() != expected_freeze_sha256:
        raise ContractError("freeze does not match reviewed SHA-256")
    try: freeze = json.loads(raw_freeze)
    except (OSError, json.JSONDecodeError) as exc: raise ContractError("freeze is invalid") from exc
    _validate_freeze(freeze)
    path = Path(bundle_path)
    if path.is_symlink() or not path.is_file(): raise ContractError("production bundle is unavailable")
    try: value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc: raise ContractError("production bundle is invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "wp7-vlm-prompt-bundle-v2": raise ContractError("production bundle schema is invalid")
    return value


def _regular_json(path: str | Path, role: str) -> tuple[object, str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ContractError(f"{role} must be a regular JSON file")
    raw = source.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{role} is invalid JSON") from exc
    return value, hashlib.sha256(raw).hexdigest()


def _resolve_internal_image_locators(
    value: object, *, locator_parent: Path,
) -> dict[str, dict[str, object]]:
    """Resolve locator paths relative to their artifact, never the caller CWD."""
    if not isinstance(value, Mapping):
        raise ContractError("image locator must be an object")
    resolved: dict[str, dict[str, object]] = {}
    for image_id, row in value.items():
        fields = {
            "path", "sha256", "source_full_frame_sha256", "crop_pixel_sha256",
            "crop_box_xyxy", "source_native_dimensions", "crop_native_dimensions",
        }
        if (
            not isinstance(image_id, str)
            or not image_id
            or not isinstance(row, Mapping)
            or set(row) != fields
            or not isinstance(row.get("path"), str)
            or not row["path"]
        ):
            raise ContractError("image locator fields are invalid")
        supplied = Path(row["path"])
        if not supplied.is_absolute() and ".." in supplied.parts:
            raise ContractError("image locator relative path escapes its artifact root")
        path = supplied if supplied.is_absolute() else locator_parent / supplied
        resolved[image_id] = {**dict(row), "path": str(path)}
    return resolved


def _assert_output_absent(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise ContractError(f"refusing to overwrite output: {output}")
    if output.parent.exists() and (
        output.parent.is_symlink() or not output.parent.is_dir()
    ):
        raise ContractError("output parent is not a real directory")
    return output


def _write_json_exclusive(path: str | Path, value: object) -> None:
    from fluidmix.model_bakeoff_runner import (
        ContractError as BakeoffContractError,
        write_json_exclusive,
    )

    try:
        write_json_exclusive(path, value)
    except BakeoffContractError as exc:
        raise ContractError(str(exc)) from exc


def _load_reviewed_freeze(
    path: str | Path, expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    value, observed = _regular_json(path, "development freeze")
    if _SHA.fullmatch(expected_sha256) is None or observed != expected_sha256:
        raise ContractError("development freeze does not match reviewed SHA-256")
    _validate_freeze(value)
    return dict(value), observed


def load_runtime_context(
    cohort_path: str | Path,
    image_locator_path: str | Path,
    tracker_receipt_paths: Sequence[str | Path],
    *,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    production: bool = False,
) -> RuntimeContext:
    """Rebuild one bundle and bind it to a reviewed development freeze."""
    freeze, freeze_sha = _load_reviewed_freeze(
        freeze_path, expected_freeze_sha256
    )
    if (
        not isinstance(tracker_receipt_paths, Sequence)
        or isinstance(tracker_receipt_paths, (str, bytes))
        or len(tracker_receipt_paths) != 3
    ):
        raise ContractError("exactly three tracker receipt files are required")
    cohort, cohort_sha = _regular_json(cohort_path, "path-free cohort")
    if not isinstance(cohort, Mapping):
        raise ContractError("path-free cohort must be an object")
    if production and cohort.get("partition") != "production":
        raise ContractError("production runtime requires the production partition")
    if not production and cohort.get("partition") == "production":
        raise ContractError("production partition requires production runtime mode")
    locators, locator_sha = _regular_json(image_locator_path, "image locator")
    locators = _resolve_internal_image_locators(
        locators, locator_parent=Path(image_locator_path).resolve().parent
    )
    tracker_values: list[object] = []
    tracker_file_sha: list[str] = []
    for path in tracker_receipt_paths:
        value, digest = _regular_json(path, "tracker receipt")
        tracker_values.append(value)
        tracker_file_sha.append(digest)
    built = build_vial1_bundle(
        cohort, locators, tracker_receipts=tracker_values, production=production
    )
    checked = _validate_tracker_cohort_receipts(tracker_values)
    expected_freeze_binding = {
        "tracker_cohort_id": checked[0]["cohort_id"],
        "tracker_vial_ids": [row["vial_id"] for row in checked],
        "tracker_receipt_sha256": [row["receipt_sha256"] for row in checked],
        "tracker_calibration_sha256": sorted(
            {row["calibration_sha256"] for row in checked}
        ),
        "tracker_recipe_artifact_sha256": sorted(
            {row["recipe_artifact_sha256"] for row in checked}
        ),
        "tracker_claim_boundary": checked[0]["claim_boundary"],
        "tracker_recipe_stages": [
            {key: stage[key] for key in _TRACKER_RECIPE_KEYS}
            for stage in checked[0]["stages"]
        ],
        "tracker_source_artifact_sha256": sorted(
            {row["source_artifact_sha256"] for row in checked}
        ),
    }
    if not production and any(
        freeze.get(key) != value for key, value in expected_freeze_binding.items()
    ):
        raise ContractError("development freeze does not bind the supplied tracker cohort")
    binding = {
        "production": production,
        "generation_config": dict(freeze["generation_config"]),
        "bundle_sha256": _digest(built.prompt_payload),
        "cohort_artifact_sha256": cohort_sha,
        "image_locator_artifact_sha256": locator_sha,
        "tracker_artifact_sha256": tracker_file_sha,
        "tracker_receipt_sha256": [row["receipt_sha256"] for row in checked],
        "deterministic_receipt_sha256": built.prompt_payload[
            "deterministic_binding"
        ]["receipt_sha256"],
    }
    return RuntimeContext(
        built=built, freeze_sha256=freeze_sha, binding=binding,
        production=production,
    )


def _prompt_and_images(
    context: RuntimeContext, repository: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    from fluidmix.model_bakeoff_runner import ImageInput

    built = context.built
    mode = MODEL_REASONING_MODES.get(repository) if repository is not None else None
    mode_prompt = {
        "schema_json": "Emit only the schema JSON object; do not emit thinking tags.",
        "native_think_answer": (
            "Use native reasoning exactly once, then the constrained answer: "
            "<think>...</think><answer>{strict JSON}</answer>."
        ),
    }.get(mode, "")
    prompt = f"{built.system_prompt}\n{mode_prompt}\n\n{built.user_prompt}"
    images = tuple(
        ImageInput(
            path=built.image_locators[row["image_id"]].path,
            sha256=built.image_locators[row["image_id"]].sha256,
            image_id=row["image_id"],
        )
        for row in built.prompt_payload["images"]
    )
    return prompt, images


def _attempt_expected_binding(
    context: RuntimeContext, repository: str,
) -> dict[str, Any]:
    prompt, images = _prompt_and_images(context, repository)
    return {
        **context.binding,
        "freeze_sha256": context.freeze_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "ordered_image_ids": [row.image_id for row in images],
        "ordered_image_sha256": [row.sha256 for row in images],
    }


def _invoke_projection(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("model invocation result is invalid")
    status = value.get("status")
    raw = value.get("response_raw")
    if status not in {"completed", "error"}:
        raise ContractError("model invocation status is invalid")
    if raw is not None and not isinstance(raw, str):
        raise ContractError("model raw response is invalid")
    response_sha = (
        hashlib.sha256(raw.encode("utf-8")).hexdigest() if raw is not None else None
    )
    if value.get("response_utf8_sha256") not in (None, response_sha):
        raise ContractError("model raw response SHA-256 differs")
    numeric_fields = (
        "elapsed_seconds", "peak_cuda_memory_allocated_bytes",
        "peak_cuda_memory_reserved_bytes", "peak_memory_bytes",
    )
    projected: dict[str, Any] = {
        "status": status,
        "response_raw": raw,
        "response_utf8_sha256": response_sha,
        "generated_token_count": value.get("generated_token_count"),
        "prompt_token_count": value.get("prompt_token_count"),
        "model_context_limit": value.get("model_context_limit"),
    }
    token_count = projected["generated_token_count"]
    if token_count is not None and (
        isinstance(token_count, bool) or not isinstance(token_count, int)
        or token_count < 0
    ):
        raise ContractError("model generated token count is invalid")
    for field in ("prompt_token_count", "model_context_limit"):
        item = projected[field]
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
        ):
            raise ContractError(f"model {field} is invalid")
    for field in numeric_fields:
        item = value.get(field)
        if item is not None:
            minimum = 0.0
            item = _finite(item, f"invoke {field}", minimum)
            if field != "elapsed_seconds" and not item.is_integer():
                raise ContractError(f"invoke {field} must be an integer")
            item = item if field == "elapsed_seconds" else int(item)
        projected[field] = item
    if projected["elapsed_seconds"] is None or projected["peak_memory_bytes"] is None:
        raise ContractError("model invocation must record latency and peak memory")
    if status == "completed" and (
        projected["prompt_token_count"] is None
        or projected["model_context_limit"] is None
    ):
        raise ContractError("completed invocation lacks prompt/context telemetry")
    error_type, error = value.get("error_type"), value.get("error")
    projected["error_type"] = error_type if isinstance(error_type, str) else None
    projected["error"] = error[:4000] if isinstance(error, str) else None
    return projected


def _validate_attempt_receipt(
    value: object, context: RuntimeContext,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("model attempt receipt is invalid")
    fields = {
        "schema_version", "artifact_type", "cohort_id", "binding", "prompt",
        "images", "model", "invoke", "prerequisite_qwen_receipt_sha256",
        "constrained_decoding", "determinism",
        "token_budget",
        "receipt_sha256",
    }
    if set(value) != fields:
        raise ContractError("model attempt receipt fields are not closed")
    if (
        value.get("schema_version") != "wp7-vlm-model-attempt-receipt-v2"
        or value.get("artifact_type") != "wp7_vlm_model_attempt_receipt"
        or value.get("cohort_id") != context.built.prompt_payload.get("cohort_id")
    ):
        raise ContractError("model attempt receipt identity is invalid")
    expected_receipt_sha = _digest(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    if value.get("receipt_sha256") != expected_receipt_sha:
        raise ContractError("model attempt receipt SHA mismatch")
    model = value.get("model")
    repository = model.get("repository") if isinstance(model, Mapping) else None
    if not isinstance(repository, str) or value.get("binding") != _attempt_expected_binding(context, repository):
        raise ContractError("model attempt receipt binding differs")
    prompt, images = _prompt_and_images(context, repository)
    expected_prompt = {
        "system_prompt_sha256": hashlib.sha256(
            context.built.system_prompt.encode("utf-8")
        ).hexdigest(),
        "user_prompt_sha256": hashlib.sha256(
            context.built.user_prompt.encode("utf-8")
        ).hexdigest(),
        "combined_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    expected_images = [
        {
            "image_id": image.image_id, "crop_sha256": image.sha256,
            "source_full_frame_sha256": metadata["source_full_frame_sha256"],
            "crop_pixel_sha256": metadata["crop_pixel_sha256"],
            "crop_box_xyxy": metadata["crop_box_xyxy"],
            "source_native_dimensions": metadata["source_native_dimensions"],
            "crop_native_dimensions": metadata["crop_native_dimensions"],
        }
        for metadata, image in zip(
            context.built.prompt_payload["images"], images, strict=True
        )
    ]
    if value.get("prompt") != expected_prompt or value.get("images") != expected_images:
        raise ContractError("model attempt prompt/image provenance differs")
    if not isinstance(model, Mapping) or set(model) != {
        "repository", "revision", "snapshot_inventory_sha256",
        "trust_remote_code", "local_files_only",
    }:
        raise ContractError("model attempt model provenance is invalid")
    pinned = {row["repository"]: row["revision"] for row in MODELS}
    if (
        pinned.get(model.get("repository")) != model.get("revision")
        or model.get("trust_remote_code") is not False
        or model.get("local_files_only") is not True
        or _SHA.fullmatch(str(model.get("snapshot_inventory_sha256"))) is None
    ):
        raise ContractError("model attempt is not an exact pinned stock runtime")
    decoder = value.get("constrained_decoding")
    if (
        not isinstance(decoder, Mapping)
        or decoder.get("backend") != "xgrammar"
        or decoder.get("version") != "0.2.8"
        or decoder.get("reasoning_mode") != MODEL_REASONING_MODES[model["repository"]]
        or decoder.get("constrained_region") != (
            "whole_response_json" if model["repository"] == MODELS[0]["repository"]
            else "answer_json_only"
        )
    ):
        raise ContractError("model attempt constrained-decoding provenance differs")
    if value.get("determinism") != {
        "seed": GENERATION_CONFIG["seed"],
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8",
    }:
        raise ContractError("model attempt determinism provenance differs")
    projected_invoke = _invoke_projection(value.get("invoke"))
    if value.get("invoke") != projected_invoke:
        raise ContractError("model attempt invocation provenance differs")
    token_budget = value.get("token_budget")
    if (
        not isinstance(token_budget, Mapping)
        or token_budget.get("requested_max_new_tokens") != 10000
        or token_budget.get("effective_max_new_tokens") != 10000
        or token_budget.get("fallback_used") is not False
        or token_budget.get("fallback_proof_receipt_sha256") is not None
        or token_budget.get("prompt_token_count") != projected_invoke.get("prompt_token_count")
        or token_budget.get("model_context_limit") != projected_invoke.get("model_context_limit")
    ):
        raise ContractError("model attempt token-budget provenance differs")
    prerequisite = value.get("prerequisite_qwen_receipt_sha256")
    if model["repository"] == MODELS[0]["repository"]:
        if prerequisite is not None:
            raise ContractError("Qwen attempt must not have a prerequisite")
    elif not isinstance(prerequisite, str) or _SHA.fullmatch(prerequisite) is None:
        raise ContractError("non-Qwen attempt lacks its Qwen prerequisite binding")
    return dict(value)


def run_one_model_attempt_from_files(
    cohort_path: str | Path,
    image_locator_path: str | Path,
    tracker_receipt_paths: Sequence[str | Path],
    *,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    repository: str,
    revision: str,
    snapshot_path: str | Path,
    output_path: str | Path,
    prerequisite_qwen_receipt_path: str | Path | None = None,
    production: bool = False,
    runtime_loader: Any = None,
    runtime_invoker: Any = None,
    constrained_decoder_builder: Any = None,
) -> dict[str, Any]:
    """Run exactly one pinned stock model and exclusively publish its raw attempt."""
    from fluidmix.model_bakeoff_runner import (
        MODEL_SPECS, invoke_stock_runtime, load_stock_runtime,
    )

    _assert_output_absent(output_path)
    context = load_runtime_context(
        cohort_path, image_locator_path, tracker_receipt_paths,
        freeze_path=freeze_path,
        expected_freeze_sha256=expected_freeze_sha256,
        production=production,
    )
    pins = {row["repository"]: row["revision"] for row in MODELS}
    if pins.get(repository) != revision:
        raise ContractError("runtime model is not one exact pinned model/revision")
    spec = MODEL_SPECS.get(repository)
    if spec is None or not spec.stock_runtime or spec.trust_remote_code:
        raise ContractError("runtime requires the stock trust_remote_code=False adapter")
    qwen_repository = MODELS[0]["repository"]
    prerequisite_sha = None
    if repository != qwen_repository:
        if prerequisite_qwen_receipt_path is None:
            raise ContractError("Cosmos/Anomaly runtime requires a valid Qwen attempt first")
        prerequisite_value, _ = _regular_json(
            prerequisite_qwen_receipt_path, "Qwen prerequisite receipt"
        )
        prerequisite = _validate_attempt_receipt(prerequisite_value, context)
        if prerequisite["model"]["repository"] != qwen_repository:
            raise ContractError("prerequisite attempt is not Qwen")
        try:
            parsed = parse_model_response(
                prerequisite["invoke"]["response_raw"], repository=qwen_repository
            )
        except ContractError as exc:
            raise ContractError("Qwen prerequisite output did not parse") from exc
        if guard_model_response(parsed, context.built.prompt_payload)["guard_status"] != "admitted":
            raise ContractError("Qwen prerequisite output failed the response guard")
        prerequisite_sha = prerequisite["receipt_sha256"]
    elif prerequisite_qwen_receipt_path is not None:
        raise ContractError("Qwen runtime must not declare a Qwen prerequisite")

    prompt, images = _prompt_and_images(context, repository)
    runtime_loader = runtime_loader or load_stock_runtime
    runtime_invoker = runtime_invoker or invoke_stock_runtime
    if constrained_decoder_builder is None:
        from fluidmix.wp7_vlm_constrained import build_logits_processor
        constrained_decoder_builder = build_logits_processor
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = GENERATION_CONFIG[
        "cublas_workspace_config"
    ]
    runtime = runtime_loader(repository, revision, snapshot_path)
    try:
        inventory = getattr(runtime, "inventory", None)
        if (
            not isinstance(inventory, Mapping)
            or _SHA.fullmatch(str(inventory.get("sha256"))) is None
        ):
            raise ContractError("stock runtime lacks snapshot inventory provenance")
        _configure_runtime_determinism(runtime)
        logits_processor, decoder_receipt = constrained_decoder_builder(
            runtime, repository=repository, response_schema=RESPONSE_SCHEMA
        )
        invoke = _invoke_projection(runtime_invoker(
            runtime,
            prompt,
            images,
            max_new_tokens=context.binding["generation_config"]["max_new_tokens"],
            logits_processor=logits_processor,
            processor_pixels=context.binding["generation_config"]["processor_pixels"],
        ))
    finally:
        del runtime
    receipt: dict[str, Any] = {
        "schema_version": "wp7-vlm-model-attempt-receipt-v2",
        "artifact_type": "wp7_vlm_model_attempt_receipt",
        "cohort_id": context.built.prompt_payload["cohort_id"],
        "binding": _attempt_expected_binding(context, repository),
        "prompt": {
            "system_prompt_sha256": hashlib.sha256(
                context.built.system_prompt.encode("utf-8")
            ).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(
                context.built.user_prompt.encode("utf-8")
            ).hexdigest(),
            "combined_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        "images": [
            {
                "image_id": image.image_id, "crop_sha256": image.sha256,
                "source_full_frame_sha256": metadata["source_full_frame_sha256"],
                "crop_pixel_sha256": metadata["crop_pixel_sha256"],
                "crop_box_xyxy": metadata["crop_box_xyxy"],
                "source_native_dimensions": metadata["source_native_dimensions"],
                "crop_native_dimensions": metadata["crop_native_dimensions"],
            }
            for metadata, image in zip(
                context.built.prompt_payload["images"], images, strict=True
            )
        ],
        "model": {
            "repository": repository,
            "revision": revision,
            "snapshot_inventory_sha256": inventory["sha256"],
            "trust_remote_code": False,
            "local_files_only": True,
        },
        "invoke": invoke,
        "constrained_decoding": decoder_receipt,
        "determinism": {
            "seed": GENERATION_CONFIG["seed"],
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cublas_workspace_config": ":4096:8",
        },
        "token_budget": {
            "requested_max_new_tokens": 10000,
            "effective_max_new_tokens": 10000,
            "fallback_used": False,
            "fallback_proof_receipt_sha256": None,
            "prompt_token_count": invoke["prompt_token_count"],
            "model_context_limit": invoke["model_context_limit"],
            "compatibility": (
                "prompt_plus_10000_lte_context"
                if invoke["prompt_token_count"] is not None
                and invoke["model_context_limit"] is not None
                and invoke["prompt_token_count"] + 10000 <= invoke["model_context_limit"]
                else "not_proven"
            ),
        },
        "prerequisite_qwen_receipt_sha256": prerequisite_sha,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    _write_json_exclusive(output_path, receipt)
    return receipt


def aggregate_model_attempts_from_files(
    cohort_path: str | Path,
    image_locator_path: str | Path,
    tracker_receipt_paths: Sequence[str | Path],
    attempt_receipt_paths: Sequence[str | Path],
    *,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    output_path: str | Path,
    production: bool = False,
) -> dict[str, Any]:
    """CPU-only validation and equal-weight aggregation of zero to three attempts."""
    _assert_output_absent(output_path)
    if (
        not isinstance(attempt_receipt_paths, Sequence)
        or isinstance(attempt_receipt_paths, (str, bytes))
        or len(attempt_receipt_paths) > 3
    ):
        raise ContractError("aggregation accepts zero to three attempt receipts")
    context = load_runtime_context(
        cohort_path, image_locator_path, tracker_receipt_paths,
        freeze_path=freeze_path,
        expected_freeze_sha256=expected_freeze_sha256,
        production=production,
    )
    attempts: list[dict[str, Any]] = []
    receipt_hashes: list[str] = []
    checked_receipts: list[dict[str, Any]] = []
    for path in attempt_receipt_paths:
        raw, _ = _regular_json(path, "model attempt receipt")
        checked = _validate_attempt_receipt(raw, context)
        checked_receipts.append(checked)
        attempts.append({
            "repository": checked["model"]["repository"],
            "revision": checked["model"]["revision"],
            "raw_response": checked["invoke"]["response_raw"],
        })
        receipt_hashes.append(checked["receipt_sha256"])
    by_hash = {row["receipt_sha256"]: row for row in checked_receipts}
    qwen_hashes = {
        digest for digest, row in by_hash.items()
        if row["model"]["repository"] == MODELS[0]["repository"]
    }
    if any(
        row["model"]["repository"] != MODELS[0]["repository"]
        and row["prerequisite_qwen_receipt_sha256"] not in qwen_hashes
        for row in by_hash.values()
    ):
        raise ContractError("non-Qwen aggregation requires its bound Qwen receipt")
    decision = aggregate_decision(context.built.prompt_payload, attempts)
    publication: dict[str, Any] = {
        "schema_version": "wp7-vlm-aggregation-publication-v1",
        "artifact_type": "wp7_vlm_aggregation_publication",
        "cohort_id": context.built.prompt_payload["cohort_id"],
        "bundle_sha256": context.binding["bundle_sha256"],
        "freeze_sha256": context.freeze_sha256,
        "attempt_receipt_sha256": receipt_hashes,
        "decision": decision,
    }
    publication["receipt_sha256"] = _digest(publication)
    _write_json_exclusive(output_path, publication)
    return publication


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the development-only WP7 VLM decision contract")
    parser.add_argument("--output", required=True); parser.add_argument("--development-input", action="append", required=True)
    parser.add_argument("--tracker-receipt", action="append", required=True)
    parser.add_argument("--materialization-receipt", required=True)
    args = parser.parse_args(argv)
    receipts = [json.loads(Path(item).read_text()) for item in args.tracker_receipt]
    create_development_freeze(
        args.output,
        args.development_input,
        tracker_receipts=receipts,
        materialization_receipt=args.materialization_receipt,
    )
    return 0

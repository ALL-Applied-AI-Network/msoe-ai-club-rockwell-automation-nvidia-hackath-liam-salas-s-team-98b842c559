"""Deterministic liquid-mask colour measurements for frozen synthetic plans."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LOCATOR_FIELDS = {"rgb_path", "rgb_sha256", "mask_path", "mask_sha256"}
_ORACLE_KEYS = {
    "synthetic_profile_id",
    "profile_id",
    "appearance_blend_alpha",
    "appearance_blend_alpha_by_vial",
    "synthetic_onset_id",
    "clean_stage_linear_rgb",
    "observed_stage_linear_rgb",
    "semantic_aov",
    "residual_particle_count",
    "residual_particle_count_by_vial",
    "residual_particle_plan",
    "residual_source_component_counts",
    "synthetic_oracle_record",
    "expected_disposition",
    "label_source",
    "render_seed",
}
_ORACLE_VALUES = {
    "none",
    "first_only",
    "first_two",
    "all_three",
    "after_t200",
    "after_t400",
    "after_t100",
}
_DECISION_CONTEXT_FIELDS = (
    "decision_id",
    "family_id",
    "batch_id",
    "cohort",
    "partition",
    "previous_sim_recipe_id",
    "current_sim_recipe_id",
    "previous_programmed_ml",
    "current_programmed_ml",
    "programmed_dispense_order",
    "vial_id",
    "vial_position",
    "decision_timing",
)


def _invalid_measurement(image_id: str, code: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "validity": "invalid",
        "limit_codes": [code],
        "rgb_sha256": None,
        "mask_sha256": None,
        "liquid_pixel_count": None,
        "lab_raw": None,
        "lab_6dp": None,
        "unit": "cie_lab_d65_2deg",
    }


def _read_bound_bytes(path_text: str, expected_hash: str, role: str) -> tuple[bytes, str] | str:
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"{role} path must be absolute")
    try:
        data = path.read_bytes()
    except OSError:
        return f"{role}_missing"
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_hash:
        return f"{role}_hash_mismatch"
    return data, digest


def measure_image(image_id: object, locator: object) -> dict[str, object]:
    """Measure one SHA-bound RGB/mask pair, failing closed on pixel evidence."""

    if not isinstance(image_id, str) or not image_id:
        raise ValueError("image_id must be a nonempty string")
    if not isinstance(locator, Mapping) or set(locator) != _LOCATOR_FIELDS:
        raise ValueError("image locator fields are malformed")
    for field in ("rgb_path", "mask_path"):
        if not isinstance(locator[field], str) or not locator[field]:
            raise ValueError(f"image locator {field} must be a nonempty string")
    for field in ("rgb_sha256", "mask_sha256"):
        if not isinstance(locator[field], str) or _SHA256.fullmatch(locator[field]) is None:
            raise ValueError(f"image locator {field} must be a lowercase SHA-256")

    rgb_bound = _read_bound_bytes(locator["rgb_path"], locator["rgb_sha256"], "rgb")
    if isinstance(rgb_bound, str):
        return _invalid_measurement(image_id, rgb_bound)
    mask_bound = _read_bound_bytes(
        locator["mask_path"], locator["mask_sha256"], "mask"
    )
    if isinstance(mask_bound, str):
        return _invalid_measurement(image_id, mask_bound)

    from PIL import Image, UnidentifiedImageError
    import numpy as np
    from skimage.color import rgb2lab

    try:
        with Image.open(BytesIO(rgb_bound[0])) as image:
            if image.format != "PNG" or image.mode != "RGB":
                return _invalid_measurement(image_id, "rgb_not_uint8_rgb_png")
            image.load()
            rgb = np.asarray(image, dtype=np.uint8).copy()
        with Image.open(BytesIO(mask_bound[0])) as image:
            if image.format != "PNG" or image.mode not in {"L", "1"}:
                return _invalid_measurement(image_id, "mask_not_binary_png")
            image.load()
            mask = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError, ValueError):
        return _invalid_measurement(image_id, "png_decode_failed")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return _invalid_measurement(image_id, "rgb_shape_invalid")
    if mask.shape != rgb.shape[:2]:
        return _invalid_measurement(image_id, "rgb_mask_shape_mismatch")
    values = set(int(value) for value in np.unique(mask))
    if not values.issubset({0, 255}):
        return _invalid_measurement(image_id, "mask_not_binary")
    selected = mask == 255
    liquid_pixels = int(np.count_nonzero(selected))
    if liquid_pixels <= 0:
        return _invalid_measurement(image_id, "mask_empty")
    lab_image = rgb2lab(rgb.astype(np.float64) / 255.0)
    descriptor = [float(np.median(lab_image[..., channel][selected])) for channel in range(3)]
    if not all(math.isfinite(value) for value in descriptor):
        return _invalid_measurement(image_id, "lab_nonfinite")
    return {
        "image_id": image_id,
        "validity": "valid",
        "limit_codes": [],
        "rgb_sha256": rgb_bound[1],
        "mask_sha256": mask_bound[1],
        "liquid_pixel_count": liquid_pixels,
        "lab_raw": descriptor,
        "lab_6dp": [f"{value:.6f}" for value in descriptor],
        "unit": "cie_lab_d65_2deg",
    }


def _reject_oracles(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in _ORACLE_KEYS:
                raise ValueError(f"model input contains forbidden oracle key: {key}")
            _reject_oracles(child)
    elif isinstance(value, list):
        for child in value:
            _reject_oracles(child)
    elif isinstance(value, str) and value in _ORACLE_VALUES:
        raise ValueError(f"model input contains forbidden oracle value: {value}")


def _euclidean(left: object, right: object) -> float:
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("Lab descriptor is malformed")
    if len(left) != 3 or len(right) != 3:
        raise ValueError("Lab descriptor must contain three channels")
    value = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right, strict=True)))
    if not math.isfinite(value):
        raise ValueError("delta E76 is nonfinite")
    return value


def _referenced_image_ids(decisions: list[object]) -> set[str]:
    referenced: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, Mapping) or not isinstance(decision.get("stages"), list):
            raise ValueError("model input decision records are malformed")
        for stage in decision["stages"]:
            if not isinstance(stage, Mapping):
                raise ValueError("model input stage record is malformed")
            target_id = stage.get("target_image_id")
            cases = stage.get("reference_cases")
            if not isinstance(target_id, str) or not target_id or not isinstance(cases, list):
                raise ValueError("model input stage references are malformed")
            referenced.add(target_id)
            for case in cases:
                if not isinstance(case, Mapping) or not isinstance(
                    case.get("candidate_peers"), list
                ):
                    raise ValueError("model input reference case is malformed")
                for peer in case["candidate_peers"]:
                    if not isinstance(peer, Mapping):
                        raise ValueError("model input candidate peer is malformed")
                    peer_id = peer.get("peer_image_id")
                    if not isinstance(peer_id, str) or not peer_id:
                        raise ValueError("model input candidate image ID is malformed")
                    referenced.add(peer_id)
    return referenced


def _invalid_case(case: Mapping[str, object], codes: list[str]) -> dict[str, object]:
    return {
        "case_id": case.get("case_id"),
        "mode": case.get("mode"),
        "alternative_id": case.get("alternative_id"),
        "disposition_source": case.get("disposition_source") is True,
        "validity": "insufficient_evidence",
        "limit_codes": codes,
        "peer_ids": None,
        "peer_origins": None,
        "peer_count": None,
        "peer_batch_ids": None,
        "peer_batch_count": None,
        "representative_peer_id": None,
        "center_lab_raw": None,
        "center_lab_6dp": None,
        "median_radius_delta_e76_raw": None,
        "max_radius_delta_e76_raw": None,
        "mad_radius_delta_e76_raw": None,
        "pairwise_diameter_delta_e76_raw": None,
        "threshold_delta_e76_raw": None,
        "threshold_delta_e76_6dp": None,
        "target_delta_e76_raw": None,
        "target_delta_e76_6dp": None,
        "stage_outlier": None,
        "unit": "delta_e76",
    }


def _evaluate_case(
    case: Mapping[str, object],
    target: Mapping[str, object],
    measurements: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if target["validity"] != "valid":
        return _invalid_case(case, ["target_measurement_invalid"])
    peers = case.get("candidate_peers")
    if not isinstance(peers, list):
        raise ValueError("candidate peers must be a list")
    candidates: dict[str, dict[str, object]] = {}
    invalid_codes: list[str] = []
    for peer in peers:
        if not isinstance(peer, Mapping):
            raise ValueError("candidate peer must be an object")
        peer_id = peer.get("peer_image_id")
        batch_id = peer.get("batch_id")
        origin = peer.get("origin")
        if (
            not isinstance(peer_id, str)
            or not peer_id
            or not isinstance(batch_id, str)
            or not batch_id
            or not isinstance(origin, str)
            or not origin
        ):
            raise ValueError("candidate peer identity is malformed")
        if peer_id in candidates:
            raise ValueError("candidate peer image IDs must be unique within a case")
        measured = measurements[peer_id]
        if measured["validity"] != "valid":
            invalid_codes.extend(
                f"candidate_measurement_invalid:{peer_id}:{code}"
                for code in measured["limit_codes"]
            )
            continue
        candidates[peer_id] = {
            "lab": measured["lab_raw"],
            "mask_sha256": measured["mask_sha256"],
            "batch_id": batch_id,
            "origin": origin,
        }
    if invalid_codes:
        return _invalid_case(case, invalid_codes)

    from .vision_dataset_plan import select_peer_reference

    reference = select_peer_reference(
        candidates, target_mask_sha256=target["mask_sha256"]
    )
    if reference["validity"] != "valid":
        return _invalid_case(case, [str(reference["limit_code"])])
    delta = _euclidean(target["lab_raw"], reference["center_lab"])
    threshold = float(reference["threshold_delta_e76"])
    outlier = delta > threshold
    return {
        "case_id": case.get("case_id"),
        "mode": case.get("mode"),
        "alternative_id": case.get("alternative_id"),
        "disposition_source": case.get("disposition_source") is True,
        "validity": "valid",
        "limit_codes": [],
        "peer_ids": reference["peer_ids"],
        "peer_origins": reference["peer_origins"],
        "peer_count": reference["peer_count"],
        "peer_batch_ids": reference["peer_batch_ids"],
        "peer_batch_count": reference["peer_batch_count"],
        "representative_peer_id": reference["representative_peer_id"],
        "center_lab_raw": reference["center_lab"],
        "center_lab_6dp": reference["serialized_6dp"]["center_lab"],
        "median_radius_delta_e76_raw": reference["median_radius_delta_e76"],
        "max_radius_delta_e76_raw": reference["max_radius_delta_e76"],
        "mad_radius_delta_e76_raw": reference["mad_radius_delta_e76"],
        "pairwise_diameter_delta_e76_raw": reference[
            "pairwise_diameter_delta_e76"
        ],
        "threshold_delta_e76_raw": threshold,
        "threshold_delta_e76_6dp": f"{threshold:.6f}",
        "target_delta_e76_raw": delta,
        "target_delta_e76_6dp": f"{delta:.6f}",
        "stage_outlier": outlier,
        "unit": "delta_e76",
    }


def _evaluate_stage(
    stage: Mapping[str, object], measurements: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    target_id = stage["target_image_id"]
    if not isinstance(target_id, str):
        raise ValueError("target image ID must be a string")
    target = measurements[target_id]
    cases_raw = stage["reference_cases"]
    if not isinstance(cases_raw, list):
        raise ValueError("reference cases must be a list")
    cases = [
        _evaluate_case(case, target, measurements)
        for case in cases_raw
        if isinstance(case, Mapping)
    ]
    if len(cases) != len(cases_raw):
        raise ValueError("reference cases must contain objects")
    primary_mode = stage.get("primary_mode")
    primaries = [
        case
        for case in cases
        if case["mode"] == primary_mode and case["disposition_source"] is True
    ]
    if len(primaries) != 1:
        raise ValueError("stage must contain exactly one primary reference case")
    primary = primaries[0]
    limit_codes: list[str] = []
    if target["validity"] != "valid":
        limit_codes.append("target_measurement_invalid")
        limit_codes.extend(str(code) for code in target["limit_codes"])
    if primary["validity"] != "valid":
        limit_codes.extend(str(code) for code in primary["limit_codes"])
    valid = target["validity"] == "valid" and primary["validity"] == "valid"
    return {
        "evidence_id": stage.get("evidence_id"),
        "measurement_id": stage.get("measurement_id"),
        "stage_id": stage.get("stage_id"),
        "stage_ordinal": stage.get("stage_ordinal"),
        "time_code": stage.get("time_code"),
        "target_image_id": target_id,
        "target_measurement": dict(target),
        "reference_cases": cases,
        "primary_mode": primary_mode,
        "validity": "valid" if valid else "insufficient_evidence",
        "limit_codes": [] if valid else list(dict.fromkeys(limit_codes)),
        "target_delta_e76_raw": primary["target_delta_e76_raw"] if valid else None,
        "target_delta_e76_6dp": primary["target_delta_e76_6dp"] if valid else None,
        "threshold_delta_e76_raw": primary["threshold_delta_e76_raw"] if valid else None,
        "threshold_delta_e76_6dp": primary["threshold_delta_e76_6dp"] if valid else None,
        "stage_outlier": primary["stage_outlier"] if valid else None,
        "unit": "delta_e76",
    }


def evaluate_model_plan(model_plan: object, image_locators: object) -> dict[str, object]:
    """Convert allowlisted model inputs and SHA-bound pixels into evidence records."""

    if not isinstance(model_plan, Mapping):
        raise TypeError("model input plan must be an object")
    _reject_oracles(model_plan)
    if model_plan.get("artifact_type") != "model_input_plan":
        raise ValueError("only model_input_plan artifacts are accepted")
    if model_plan.get("real_data_used") is not False:
        raise ValueError("this synthetic measurement path requires real_data_used=false")
    decisions_raw = model_plan.get("decision_records")
    if not isinstance(decisions_raw, list) or not decisions_raw:
        raise ValueError("model input plan decision_records must be a nonempty list")
    if (
        not isinstance(image_locators, Mapping)
        or set(image_locators) != {"schema_version", "data_origin", "images"}
        or image_locators.get("schema_version") != 1
        or image_locators.get("data_origin") != "twin"
        or not isinstance(image_locators.get("images"), Mapping)
    ):
        raise ValueError("image locator artifact must be a synthetic twin locator map")
    _reject_oracles(image_locators)
    referenced = _referenced_image_ids(decisions_raw)
    locator_images = image_locators["images"]
    assert isinstance(locator_images, Mapping)
    if any(not isinstance(image_id, str) or not image_id for image_id in locator_images):
        raise ValueError("image locator IDs must be nonempty strings")
    if set(locator_images) != referenced:
        missing = sorted(referenced.difference(locator_images))
        extra = sorted(set(locator_images).difference(referenced))
        raise ValueError(f"image locator IDs differ from model plan: missing={missing!r}, extra={extra!r}")
    measurements = {
        image_id: measure_image(image_id, locator_images[image_id])
        for image_id in sorted(referenced)
    }

    decisions: list[dict[str, object]] = []
    for decision in decisions_raw:
        assert isinstance(decision, Mapping)
        missing_context = [field for field in _DECISION_CONTEXT_FIELDS if field not in decision]
        if missing_context:
            raise ValueError(f"decision context fields are missing: {missing_context!r}")
        stages_raw = decision["stages"]
        if not isinstance(stages_raw, list) or len(stages_raw) != 3:
            raise ValueError("each vial decision must contain exactly three stages")
        stages = [_evaluate_stage(stage, measurements) for stage in stages_raw]
        if sorted(stage["stage_ordinal"] for stage in stages) != [1, 2, 3]:
            raise ValueError("vial decision stage ordinals must be exactly 1, 2, and 3")
        valid_stages = [stage for stage in stages if stage["validity"] == "valid"]
        if any(stage["stage_outlier"] is True for stage in valid_stages):
            disposition = "discard"
            limit_codes: list[str] = []
        elif len(valid_stages) == 3 and all(
            stage["stage_outlier"] is False for stage in valid_stages
        ):
            disposition = "keep"
            limit_codes = []
        else:
            disposition = "insufficient_evidence"
            limit_codes = list(
                dict.fromkeys(
                    code
                    for stage in stages
                    if stage["validity"] != "valid"
                    for code in stage["limit_codes"]
                )
            )
        context = {field: decision[field] for field in _DECISION_CONTEXT_FIELDS}
        decisions.append(
            {
                **context,
                "stages": stages,
                "disposition": disposition,
                "explanation_status": {
                    "discard": "DIFFERENCE",
                    "keep": "NO_DIFFERENCE",
                    "insufficient_evidence": "UNKNOWN",
                }[disposition],
                "disposition_source": "deterministic_pipeline_primary_reference_evidence",
                "limit_codes": limit_codes,
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "measured_evidence",
        "config_id": model_plan.get("config_id"),
        "measurement_contract": {
            "implementation": "skimage.color.rgb2lab",
            "input_encoding": "uint8_srgb_divide_255",
            "statistic": "coordinatewise_median_lab_under_binary_liquid_mask",
            "distance": "delta_e76",
            "primary_mode": "all_reference_batches",
        },
        "image_measurements": [measurements[key] for key in sorted(measurements)],
        "decision_records": decisions,
        "denominators": {
            "image_measurements": len(measurements),
            "vial_decisions": len(decisions),
            "valid_vial_decisions": sum(
                decision["disposition"] != "insufficient_evidence"
                for decision in decisions
            ),
            "abstained_vial_decisions": sum(
                decision["disposition"] == "insufficient_evidence"
                for decision in decisions
            ),
        },
        "data_origin": "twin",
        "real_data_used": False,
        "claim_boundary": "Synthetic colour-distance evidence only; no physical carryover, defect, cause, volume, plant-action, or sim-to-real claim.",
    }


def _load_strict_json(path: Path, role: str) -> tuple[object, str]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{role} is unavailable") from error
    if len(data) > 64 * 1024 * 1024:
        raise ValueError(f"{role} exceeds the size bound")

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"{role} contains nonfinite JSON value {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not strict UTF-8 JSON") from error
    return payload, hashlib.sha256(data).hexdigest()


def _publish_json_no_clobber(output: Path, payload: object) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"measured evidence output already exists: {output}")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("measured evidence output parent must be a directory")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = parent / f".{output.name}.tmp-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    published = False
    try:
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output, follow_symlinks=False)
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not published:
        raise OSError("measured evidence output was not published")


def write_measured_evidence(
    model_plan_path: str | os.PathLike[str],
    locator_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Evaluate two separate allowlisted artifacts and publish one new record."""

    plan, plan_sha256 = _load_strict_json(Path(model_plan_path), "model input plan")
    locators, locator_sha256 = _load_strict_json(
        Path(locator_path), "image locator artifact"
    )
    result = evaluate_model_plan(plan, locators)
    result["source_model_input_sha256"] = plan_sha256
    result["source_image_locators_sha256"] = locator_sha256
    _publish_json_no_clobber(Path(output_path), result)
    return result

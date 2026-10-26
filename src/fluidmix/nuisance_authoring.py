"""Deterministic, uncalibrated lighting nuisance plans for F2 twin renders."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re


MAPPING_VERSION = "f2-measured-percentile-to-existing-lights-v1"
REQUIRED_METRICS = (
    "luma",
    "shadow_centroid_x",
    "shadow_centroid_y",
    "shadow_edge_angle_degrees",
    "shadow_area_fraction",
    "shadow_contrast",
    "shadow_penumbra_10_90_px",
)
LIGHT_SPECS: dict[str, dict[str, object]] = {
    "/FillStation/FillStation_LocalRoot/V2_base_reflection/V2_base_reflection": {
        "type_name": "RectLight",
        "transform_path": "/FillStation/FillStation_LocalRoot/V2_base_reflection",
        "weights": (-0.75, 0.25, 0.30, 0.40),
    },
    "/FillStation/FillStation_LocalRoot/Fill_right_reflection/Fill_right_reflection": {
        "type_name": "RectLight",
        "transform_path": "/FillStation/FillStation_LocalRoot/Fill_right_reflection",
        "weights": (0.75, 0.50, 0.20, 0.70),
    },
    "/FillStation/FillStation_LocalRoot/Fill_front_softbox/Fill_front_softbox": {
        "type_name": "RectLight",
        "transform_path": "/FillStation/FillStation_LocalRoot/Fill_front_softbox",
        "weights": (1.00, -0.80, 0.50, 1.00),
    },
    "/FillStation/FillStation_LocalRoot/Fill_overhead_panel/Fill_overhead_panel": {
        "type_name": "RectLight",
        "transform_path": "/FillStation/FillStation_LocalRoot/Fill_overhead_panel",
        "weights": (0.30, 0.20, 1.00, 0.50),
    },
    "/FillStation/FillStation_LocalRoot/V2_guard_bounce/V2_guard_bounce": {
        "type_name": "RectLight",
        "transform_path": "/FillStation/FillStation_LocalRoot/V2_guard_bounce",
        "weights": (-0.40, -0.50, 0.20, 0.60),
    },
    "/FillStation/env_light": {
        "type_name": "DomeLight",
        "transform_path": None,
        "weights": None,
    },
}
EXPECTED_LIGHT_PATHS = tuple(sorted(LIGHT_SPECS))
_CASE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# Parent-Xform values use the asset's local units. These small bounds are a
# synthetic stress transform, not a calibration from image pixels to metres.
FROZEN_SYNTHETIC_LIMITS = {
    "translation_xy_asset_units": 0.75,
    "translation_z_asset_units": 0.25,
    "rotation_x_degrees": 6.0,
    "rotation_y_degrees": 4.0,
    "rotation_z_degrees": 10.0,
}


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{name} must be a finite{' positive' if positive else ''} number")
    return result


def _vec3(value: object, name: str) -> list[float]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three finite numbers")
    try:
        components = list(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain three finite numbers") from None
    if len(components) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    return [_finite(component, name) for component in components]


def _percentile_bounds(binding: Mapping[str, object]) -> dict[str, dict[str, float]]:
    raw = binding.get("percentile_bounds")
    if not isinstance(raw, Mapping):
        raise ValueError("nuisance binding percentile bounds are missing")
    result: dict[str, dict[str, float]] = {}
    for metric in REQUIRED_METRICS:
        values = raw.get(metric)
        if not isinstance(values, Mapping):
            raise ValueError(f"nuisance binding metric {metric} is missing")
        row = {name: _finite(values.get(name), f"{metric}.{name}") for name in ("p05", "p50", "p95")}
        if not row["p05"] < row["p50"] < row["p95"]:
            raise ValueError(f"nuisance binding metric {metric} is not strictly ordered")
        result[metric] = row
    return result


def _unit(case_id: str, seed: int, metric: str) -> float:
    payload = f"{MAPPING_VERSION}\0{case_id}\0{seed}\0{metric}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / float(2**64 - 1)


def _sample(bounds: Mapping[str, float], unit: float) -> float:
    if unit <= 0.5:
        return float(bounds["p05"]) + (float(bounds["p50"]) - float(bounds["p05"])) * (unit * 2.0)
    return float(bounds["p50"]) + (float(bounds["p95"]) - float(bounds["p50"])) * ((unit - 0.5) * 2.0)


def _normalized(sample: float, bounds: Mapping[str, float]) -> float:
    median = float(bounds["p50"])
    denominator = median - float(bounds["p05"]) if sample <= median else float(bounds["p95"]) - median
    return (sample - median) / denominator


def _validated_inventory(value: Mapping[str, object]) -> dict[str, dict[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_LIGHT_PATHS):
        raise ValueError("light inventory must contain exactly the reviewed existing lights")
    result: dict[str, dict[str, object]] = {}
    for path in EXPECTED_LIGHT_PATHS:
        raw = value[path]
        spec = LIGHT_SPECS[path]
        if not isinstance(raw, Mapping) or raw.get("type_name") != spec["type_name"]:
            raise ValueError(f"light inventory type mismatch for {path}")
        row: dict[str, object] = {
            "type_name": str(raw["type_name"]),
            "intensity": _finite(raw.get("intensity"), f"{path} intensity", positive=True),
            "exposure": _finite(raw.get("exposure"), f"{path} exposure"),
            "transform_path": raw.get("transform_path"),
            "translate": raw.get("translate"),
            "rotate_xyz_degrees": raw.get("rotate_xyz_degrees"),
        }
        if spec["transform_path"] is None:
            if any(row[name] is not None for name in ("transform_path", "translate", "rotate_xyz_degrees")):
                raise ValueError(f"DomeLight {path} must remain transform-unmodified")
        else:
            if row["transform_path"] != spec["transform_path"]:
                raise ValueError(f"light inventory transform path mismatch for {path}")
            row["translate"] = _vec3(row["translate"], f"{path} translate")
            row["rotate_xyz_degrees"] = _vec3(
                row["rotate_xyz_degrees"], f"{path} rotate_xyz_degrees"
            )
        result[path] = row
    return result


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def build_nuisance_authoring_plan(
    binding: Mapping[str, object],
    *,
    case_id: str,
    seed: int,
    light_inventory: Mapping[str, object],
) -> dict[str, object]:
    """Map measured percentiles to bounded existing-light session overrides."""

    if not isinstance(binding, Mapping) or binding.get("status") != "measured":
        raise ValueError("a measured nuisance binding is required")
    receipt_sha256 = binding.get("receipt_sha256")
    if not isinstance(receipt_sha256, str) or _SHA256.fullmatch(receipt_sha256) is None:
        raise ValueError("nuisance binding receipt SHA-256 is invalid")
    if not isinstance(case_id, str) or _CASE_ID.fullmatch(case_id) is None:
        raise ValueError("case_id must be a bounded opaque identifier")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    bounds = _percentile_bounds(binding)
    inventory = _validated_inventory(light_inventory)
    sampled = {
        metric: _sample(bounds[metric], _unit(case_id, seed, metric))
        for metric in REQUIRED_METRICS
    }
    normalized = {
        metric: _normalized(sampled[metric], bounds[metric]) for metric in REQUIRED_METRICS
    }
    gain = sampled["luma"] / bounds["luma"]["p50"]
    intensity_scale = math.sqrt(gain)
    exposure_delta = 0.5 * math.log2(gain)
    authored: dict[str, dict[str, object]] = {}
    for path in EXPECTED_LIGHT_PATHS:
        source = inventory[path]
        spec = LIGHT_SPECS[path]
        row: dict[str, object] = {
            "type_name": source["type_name"],
            "intensity": float(source["intensity"]) * intensity_scale,
            "exposure": float(source["exposure"]) + exposure_delta,
            "transform_path": source["transform_path"],
            "translate": None,
            "rotate_xyz_degrees": None,
        }
        weights = spec["weights"]
        if weights is not None:
            wx, wy, wz, wr = (float(value) for value in weights)
            translate = [float(value) for value in source["translate"]]
            rotate = [float(value) for value in source["rotate_xyz_degrees"]]
            row["translate"] = [
                translate[0]
                + normalized["shadow_centroid_x"]
                * FROZEN_SYNTHETIC_LIMITS["translation_xy_asset_units"]
                * wx,
                translate[1]
                + normalized["shadow_centroid_y"]
                * FROZEN_SYNTHETIC_LIMITS["translation_xy_asset_units"]
                * wy,
                translate[2]
                + 0.5
                * (
                    normalized["shadow_area_fraction"]
                    + normalized["shadow_penumbra_10_90_px"]
                )
                * FROZEN_SYNTHETIC_LIMITS["translation_z_asset_units"]
                * wz,
            ]
            row["rotate_xyz_degrees"] = [
                rotate[0] + normalized["shadow_penumbra_10_90_px"] * FROZEN_SYNTHETIC_LIMITS["rotation_x_degrees"] * wr,
                rotate[1] + normalized["shadow_contrast"] * FROZEN_SYNTHETIC_LIMITS["rotation_y_degrees"] * wr,
                rotate[2] + normalized["shadow_edge_angle_degrees"] * FROZEN_SYNTHETIC_LIMITS["rotation_z_degrees"] * wr,
            ]
        authored[path] = row
    plan: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "f2_nuisance_authoring_plan",
        "mapping_version": MAPPING_VERSION,
        "mapping_status": "uncalibrated_synthetic_nuisance_mapping",
        "real_mapping_claimed": False,
        "chemistry_validated": False,
        "excluded_from_model_inputs": True,
        "excluded_from_deploy_inputs": True,
        "provenance_only": {"case_id": case_id, "seed": seed},
        "receipt_path": binding.get("receipt_path"),
        "receipt_sha256": receipt_sha256,
        "sampled_measured_metrics": sampled,
        "measured_percentile_bounds": bounds,
        "normalized_measured_coordinates": normalized,
        "frozen_synthetic_limits": dict(FROZEN_SYNTHETIC_LIMITS),
        "gain_split": {
            "target_relative_luma_coordinate": gain,
            "intensity_scale": intensity_scale,
            "exposure_delta_stops": exposure_delta,
            "identity": "intensity_scale * 2**exposure_delta_stops == target_relative_luma_coordinate",
        },
        "allowed_authored_properties": [
            "inputs:intensity",
            "inputs:exposure",
            "xformOp:translate",
            "xformOp:rotateXYZ",
        ],
        "light_authoring": authored,
        "limits": [
            "measured percentiles condition bounded synthetic variation only",
            "pixel shadow metrics are not calibrated to asset-space transforms",
            "no real-light, photometric, physical-fluid, or chemistry mapping is claimed",
            "camera and non-light geometry are fixed",
        ],
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _valid_prim(prim: object, path: str) -> object:
    if prim is None:
        raise ValueError(f"required light prim is missing: {path}")
    is_valid = getattr(prim, "IsValid", None)
    if callable(is_valid) and not bool(is_valid()):
        raise ValueError(f"required light prim is invalid: {path}")
    return prim


def _valid_attr(prim: object, name: str, path: str) -> object:
    get_attribute = getattr(prim, "GetAttribute", None)
    if not callable(get_attribute):
        raise ValueError(f"required light attribute is unavailable: {path}.{name}")
    attr = get_attribute(name)
    if attr is None:
        raise ValueError(f"required light attribute is missing: {path}.{name}")
    is_valid = getattr(attr, "IsValid", None)
    if callable(is_valid) and not bool(is_valid()):
        raise ValueError(f"required light attribute is invalid: {path}.{name}")
    get_value = getattr(attr, "Get", None)
    if not callable(get_value):
        raise ValueError(f"required light attribute readback is unavailable: {path}.{name}")
    return attr


def read_existing_light_inventory(stage: object) -> dict[str, dict[str, object]]:
    """Read the reviewed light prims and required source opinions from a live stage.

    This is intentionally a strict preflight.  It does not create prims or attributes;
    missing source properties are an error before any session-layer write is attempted.
    """

    get_prim = getattr(stage, "GetPrimAtPath", None)
    if not callable(get_prim):
        raise ValueError("stage cannot read light prims")
    inventory: dict[str, dict[str, object]] = {}
    for path in EXPECTED_LIGHT_PATHS:
        spec = LIGHT_SPECS[path]
        prim = _valid_prim(get_prim(path), path)
        get_type_name = getattr(prim, "GetTypeName", None)
        type_name = str(get_type_name()) if callable(get_type_name) else ""
        if type_name != spec["type_name"]:
            raise ValueError(
                f"light inventory type mismatch for {path}: expected {spec['type_name']}, got {type_name}"
            )
        intensity_attr = _valid_attr(prim, "inputs:intensity", path)
        exposure_attr = _valid_attr(prim, "inputs:exposure", path)
        get_intensity = getattr(intensity_attr, "Get")
        get_exposure = getattr(exposure_attr, "Get")
        row: dict[str, object] = {
            "type_name": type_name,
            "intensity": _finite(get_intensity(), f"{path} intensity", positive=True),
            "exposure": _finite(get_exposure(), f"{path} exposure"),
            "transform_path": spec["transform_path"],
            "translate": None,
            "rotate_xyz_degrees": None,
        }
        transform_path = spec["transform_path"]
        if transform_path is not None:
            transform_prim = _valid_prim(get_prim(transform_path), transform_path)
            translate_attr = _valid_attr(transform_prim, "xformOp:translate", transform_path)
            rotate_attr = _valid_attr(transform_prim, "xformOp:rotateXYZ", transform_path)
            row["translate"] = _vec3(
                getattr(translate_attr, "Get")(), f"{transform_path} translate"
            )
            row["rotate_xyz_degrees"] = _vec3(
                getattr(rotate_attr, "Get")(), f"{transform_path} rotate_xyz_degrees"
            )
        inventory[path] = row
    return inventory


def _set_and_readback(attr: object, value: object, path: str) -> object:
    set_value = getattr(attr, "Set", None)
    if not callable(set_value):
        raise ValueError(f"required light attribute cannot be authored: {path}")
    result = set_value(value)
    if result is False:
        raise ValueError(f"required light attribute authoring failed: {path}")
    get_value = getattr(attr, "Get", None)
    if not callable(get_value):
        raise ValueError(f"required light attribute readback is unavailable: {path}")
    readback = get_value()
    if readback is None:
        raise ValueError(f"required light attribute readback is missing: {path}")
    return readback


def _serial_light_row(row: Mapping[str, object]) -> dict[str, object]:
    result = {
        "type_name": str(row["type_name"]),
        "intensity": _finite(row["intensity"], "light intensity", positive=True),
        "exposure": _finite(row["exposure"], "light exposure"),
        "transform_path": row["transform_path"],
        "translate": None,
        "rotate_xyz_degrees": None,
    }
    if row["transform_path"] is not None:
        result["translate"] = _vec3(row["translate"], "light translate")
        result["rotate_xyz_degrees"] = _vec3(
            row["rotate_xyz_degrees"], "light rotate_xyz_degrees"
        )
    return result


def apply_nuisance_session_layer(
    stage: object,
    usd: object,
    binding: Mapping[str, object],
    *,
    case_id: str,
    seed: int,
) -> dict[str, object]:
    """Author one deterministic light-only nuisance case into a USD session layer.

    The full stage inventory is validated before entering the edit context.  The
    function never creates prims or attributes, and returns only provenance/readback
    metadata; callers must not pass this nuisance record into model inputs.
    """

    get_session_layer = getattr(stage, "GetSessionLayer", None)
    if not callable(get_session_layer):
        raise ValueError("stage cannot provide a session layer")
    session_layer = get_session_layer()
    if session_layer is None:
        raise ValueError("required USD session layer is missing")
    inventory = read_existing_light_inventory(stage)
    plan = build_nuisance_authoring_plan(
        binding,
        case_id=case_id,
        seed=seed,
        light_inventory=inventory,
    )
    authored = plan["light_authoring"]
    if not isinstance(authored, Mapping):
        raise ValueError("nuisance plan light authoring is missing")
    get_prim = getattr(stage, "GetPrimAtPath")
    authored_rows: dict[str, dict[str, object]] = {}
    readback_rows: dict[str, dict[str, object]] = {}
    edit_context = getattr(usd, "EditContext", None)
    if not callable(edit_context):
        raise ValueError("USD session edit context is unavailable")
    with edit_context(stage, session_layer):
        for path in EXPECTED_LIGHT_PATHS:
            spec = LIGHT_SPECS[path]
            source = authored.get(path)
            if not isinstance(source, Mapping):
                raise ValueError(f"nuisance plan light is missing: {path}")
            light_prim = _valid_prim(get_prim(path), path)
            intensity_attr = _valid_attr(light_prim, "inputs:intensity", path)
            exposure_attr = _valid_attr(light_prim, "inputs:exposure", path)
            authored_row = _serial_light_row(source)
            intensity_readback = _set_and_readback(
                intensity_attr,
                authored_row["intensity"],
                f"{path}.inputs:intensity",
            )
            exposure_readback = _set_and_readback(
                exposure_attr,
                authored_row["exposure"],
                f"{path}.inputs:exposure",
            )
            readback_row: dict[str, object] = {
                "type_name": spec["type_name"],
                "intensity": _finite(
                    intensity_readback, f"{path} intensity readback", positive=True
                ),
                "exposure": _finite(exposure_readback, f"{path} exposure readback"),
                "transform_path": spec["transform_path"],
                "translate": None,
                "rotate_xyz_degrees": None,
            }
            transform_path = spec["transform_path"]
            if transform_path is not None:
                transform_prim = _valid_prim(get_prim(transform_path), transform_path)
                translate_attr = _valid_attr(
                    transform_prim, "xformOp:translate", transform_path
                )
                rotate_attr = _valid_attr(
                    transform_prim, "xformOp:rotateXYZ", transform_path
                )
                translate_readback = _set_and_readback(
                    translate_attr,
                    authored_row["translate"],
                    f"{transform_path}.xformOp:translate",
                )
                rotate_readback = _set_and_readback(
                    rotate_attr,
                    authored_row["rotate_xyz_degrees"],
                    f"{transform_path}.xformOp:rotateXYZ",
                )
                readback_row["translate"] = _vec3(
                    translate_readback, f"{transform_path} translate readback"
                )
                readback_row["rotate_xyz_degrees"] = _vec3(
                    rotate_readback, f"{transform_path} rotate_xyz_degrees readback"
                )
            authored_rows[path] = authored_row
            readback_rows[path] = readback_row
    if authored_rows != readback_rows:
        raise ValueError("nuisance light authored/readback mismatch")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "f2_nuisance_session_authoring_receipt",
        "status": "applied",
        "session_layer_only": True,
        "case_id": case_id,
        "seed": seed,
        "source_receipt_sha256": plan["receipt_sha256"],
        "source_sha256": plan["receipt_sha256"],
        "mapping_version": plan["mapping_version"],
        "provenance_only": dict(plan["provenance_only"]),
        "mapping": {
            "version": plan["mapping_version"],
            "status": plan["mapping_status"],
            "real_mapping_claimed": plan["real_mapping_claimed"],
            "measured_percentile_bounds": plan["measured_percentile_bounds"],
            "sampled_measured_metrics": plan["sampled_measured_metrics"],
            "frozen_synthetic_limits": plan["frozen_synthetic_limits"],
        },
        "authored_light_properties": authored_rows,
        "readback_light_properties": readback_rows,
        "authored_transforms": {
            path: {
                "transform_path": row["transform_path"],
                "translate": row["translate"],
                "rotate_xyz_degrees": row["rotate_xyz_degrees"],
            }
            for path, row in authored_rows.items()
        },
        "readback_transforms": {
            path: {
                "transform_path": row["transform_path"],
                "translate": row["translate"],
                "rotate_xyz_degrees": row["rotate_xyz_degrees"],
            }
            for path, row in readback_rows.items()
        },
        "allowed_authored_properties": plan["allowed_authored_properties"],
        "shadows": {
            "existing_light_position_orientation_used": True,
            "shadow_parameters_authored": False,
        },
        "excluded_from_model_inputs": True,
        "model_inputs": {},
        "camera_pose_unchanged": True,
        "non_light_scene_unchanged": True,
        "physics_recipe_particles_labels_unchanged": True,
        "input_evidence_unchanged": True,
        "forbidden_mutations": [
            "FillCam/camera pose",
            "vial and non-light scene geometry",
            "colliders, physics and particles",
            "recipe and labels",
            "input evidence",
        ],
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


author_nuisance_session_layer = apply_nuisance_session_layer


__all__ = [
    "EXPECTED_LIGHT_PATHS",
    "FROZEN_SYNTHETIC_LIMITS",
    "LIGHT_SPECS",
    "MAPPING_VERSION",
    "REQUIRED_METRICS",
    "apply_nuisance_session_layer",
    "author_nuisance_session_layer",
    "build_nuisance_authoring_plan",
    "read_existing_light_inventory",
]

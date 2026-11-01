"""Integrity and look evidence for the first Revision 2 train chunk.

This module reads synthetic train renders only.  It never opens calibration or
sealed-test images, and it refuses non-train manifests.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np
from PIL import Image, ImageDraw


EXPECTED_FRAME_COUNT = 100
DEFAULT_ANALYSIS_MODE = "revision2-batch-100"
V4_LOOK_ANALYSIS_MODE = "v4-look-60"
V4_LOOK_EXPECTED_FRAME_COUNT = 60
EXPECTED_PROFILE = "real-look-v2"
EXPECTED_RENDERER = "PathTracing"
EXPECTED_SUBFRAMES = 32
EXPECTED_RESOLUTION = [640, 480]
LIQUID_ID = 1
ALLOWED_MASK_IDS = {0, 1, 2, 3, 255}
SATURATED_RANGE = (0.30, 0.85)
DARKEST_TAIL_RANGE = (0.05, 0.09)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INDEPENDENT_CHECKER_PATH = REPOSITORY_ROOT / "scripts" / "check_fill_level_capture.py"
EXPECTED_GLASS_IOR = 1.46
EXPECTED_GLASS_F0 = (
    (EXPECTED_GLASS_IOR - 1.0) / (EXPECTED_GLASS_IOR + 1.0)
) ** 2
LABEL_IDS = {
    "empty": 0,
    "liquid": 1,
    "foam": 2,
    "blocked": 3,
    "ignore": 255,
}
EXPECTED_GLASS_PARAMS = {
    "diffuse_rgb": [0.92, 0.96, 0.98],
    "opacity": 0.14,
    "roughness": 0.08,
    "ior": EXPECTED_GLASS_IOR,
    "specular_rgb": [EXPECTED_GLASS_F0] * 3,
    "use_specular_workflow": 1.0,
    "metallic": 0.0,
}
V4_LOOK_PINNED_HASHES = {
    "renderer_script_sha256": (
        "8e23ece14ed5c58599981c59589d4c9f0484703fc629ad760333e0d6f09025cd"
    ),
    "variation_module_sha256": (
        "71aea9f84e7b6c84e7ed807d3784891715e3ddbeac5dea8f17ed291d9f52a182"
    ),
    "checker_sha256": (
        "a03442a7da4cde31dee084ffbe3375f0574b370a78829c01976d77f49246e966"
    ),
    "camera_match_sha256": (
        "aeee0c5ef7e5b8b88678356f30de4d14a3126da70e4eb5c84711c674bfafc2f8"
    ),
}
V4_LOOK_VALUE_REFERENCE_RANGE = (0.09, 0.19)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_with_independent_checker(
    manifest_path,
    expected_seeds,
    *,
    expected_role="train",
    expected_checker_sha256=None,
):
    """Run the repository checker and keep only deterministic evidence fields."""
    checker_path = INDEPENDENT_CHECKER_PATH.resolve()
    if not checker_path.is_file():
        raise ValueError(
            f"independent capture checker does not exist: {checker_path}"
        )
    checker_hash = file_sha256(checker_path)
    if expected_checker_sha256 is not None:
        expected_checker_sha256 = _sha(
            expected_checker_sha256, "expected checker_sha256"
        )
        if checker_hash != expected_checker_sha256:
            raise ValueError(
                "independent capture checker does not match the pinned v4-look hash"
            )
    specification = importlib.util.spec_from_file_location(
        "_fluidmix_fill_level_capture_checker", checker_path
    )
    if specification is None or specification.loader is None:
        raise ValueError(
            f"could not load independent capture checker: {checker_path}"
        )
    checker = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(checker)
    except Exception as error:
        raise ValueError(
            f"could not import independent capture checker: {error}"
        ) from error
    validator = getattr(checker, "validate_manifest", None)
    if not callable(validator):
        raise ValueError("independent capture checker has no validate_manifest API")
    try:
        raw_report = validator(Path(manifest_path).resolve())
    except Exception as error:
        raise ValueError(
            f"independent capture checker raised an exception: {error}"
        ) from error
    if not isinstance(raw_report, dict):
        raise ValueError("independent capture checker returned no report object")
    records = raw_report.get("records")
    record_count = raw_report.get("record_count")
    expected_seeds = tuple(expected_seeds)
    if not isinstance(records, list):
        raise ValueError("independent capture checker report has no record list")
    if record_count != len(expected_seeds) or len(records) != len(expected_seeds):
        raise ValueError(
            "independent capture checker did not report exactly "
            f"{len(expected_seeds)} records"
        )
    errors = raw_report.get("errors")
    if not isinstance(errors, list):
        raise ValueError("independent capture checker report has no error list")
    if raw_report.get("passed") is not True or errors:
        detail = "; ".join(str(error) for error in errors[:5])
        if len(errors) > 5:
            detail += f"; ... ({len(errors) - 5} more)"
        raise ValueError(
            "independent capture checker did not pass"
            + (f": {detail}" if detail else "")
        )
    normalized_records = []
    for offset, (record, expected_seed) in enumerate(
        zip(records, expected_seeds), 1
    ):
        if not isinstance(record, dict):
            raise ValueError(
                f"independent capture checker record {offset - 1} is not an object"
            )
        if record.get("errors") != []:
            raise ValueError(
                f"independent capture checker record {offset - 1} contains errors"
            )
        if (
            record.get("index") != offset
            or record.get("seed") != expected_seed
            or record.get("role") != expected_role
        ):
            raise ValueError(
                "independent capture checker record identity does not match the "
                f"expected train chunk at offset {offset - 1}"
            )
        label_counts = record.get("label_counts")
        if not isinstance(label_counts, dict):
            raise ValueError(
                f"independent capture checker record {offset - 1} lacks label counts"
            )
        normalized_records.append(
            {
                "index": offset,
                "seed": expected_seed,
                "role": expected_role,
                "label_counts": label_counts,
                "errors": [],
            }
        )
    return {
        "checker_path": "scripts/check_fill_level_capture.py",
        "checker_sha256": checker_hash,
        "status": "passed",
        "passed": True,
        "record_count": record_count,
        "report": {
            "schema_version": raw_report.get("schema_version"),
            "record_count": record_count,
            "errors": [],
            "passed": True,
            "records": normalized_records,
        },
    }


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _read_manifest(path):
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read train manifest {path}: {error}") from error
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"train manifest line {line_number} is invalid JSON: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"train manifest line {line_number} is not an object")
        records.append(record)
    if not records:
        raise ValueError("train manifest contains no records")
    return records


def _sha(value, description):
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value.lower()):
        raise ValueError(f"{description} must be a SHA-256")
    return value.lower()


def _analysis_contract(analysis_mode):
    if analysis_mode == DEFAULT_ANALYSIS_MODE:
        return {
            "mode": DEFAULT_ANALYSIS_MODE,
            "expected_frame_count": EXPECTED_FRAME_COUNT,
            "pinned_hashes": None,
        }
    if analysis_mode == V4_LOOK_ANALYSIS_MODE:
        return {
            "mode": V4_LOOK_ANALYSIS_MODE,
            "expected_frame_count": V4_LOOK_EXPECTED_FRAME_COUNT,
            "pinned_hashes": dict(V4_LOOK_PINNED_HASHES),
        }
    raise ValueError(
        "analysis_mode must be revision2-batch-100 or v4-look-60"
    )


def _finite(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _safe_output(root, descriptor, description):
    if not isinstance(descriptor, dict):
        raise ValueError(f"{description} descriptor must be an object")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{description} path must be a non-empty string")
    if Path(relative).is_absolute():
        raise ValueError(f"{description} path must be relative to the chunk root")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} path escapes the chunk root") from error
    if not path.is_file():
        raise ValueError(f"{description} file does not exist: {relative}")
    expected = _sha(descriptor.get("sha256"), f"{description} hash")
    if file_sha256(path) != expected:
        raise ValueError(f"{description} SHA-256 mismatch")
    return path


def _rgb_hsv(rgb):
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    maximum = values.max(axis=-1)
    minimum = values.min(axis=-1)
    difference = maximum - minimum
    saturation = np.divide(
        difference,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    hue = np.zeros_like(maximum)
    active = difference > 0
    red = active & (maximum == values[..., 0])
    green = active & (maximum == values[..., 1])
    blue = active & (maximum == values[..., 2])
    hue[red] = (
        (values[..., 1][red] - values[..., 2][red]) / difference[red]
    ) % 6
    hue[green] = (
        (values[..., 2][green] - values[..., 0][green]) / difference[green] + 2
    )
    hue[blue] = (
        (values[..., 0][blue] - values[..., 1][blue]) / difference[blue] + 4
    )
    hue /= 6.0
    return np.stack((hue, saturation, maximum), axis=-1)


def _circular_hue_mean(hues, weights):
    hues = np.asarray(hues, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if hues.size == 0 or float(weights.sum()) <= 1e-12:
        return None
    radians = 2.0 * np.pi * hues
    angle = math.atan2(
        float(np.sum(np.sin(radians) * weights)),
        float(np.sum(np.cos(radians) * weights)),
    )
    return float((angle / (2.0 * np.pi)) % 1.0)


def liquid_hsv_metrics(rgb, liquid_mask):
    rgb = np.asarray(rgb)
    selected = np.asarray(liquid_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be an HxWx3 uint8 array")
    if selected.shape != rgb.shape[:2]:
        raise ValueError("liquid mask dimensions differ from RGB")
    pixel_count = int(np.count_nonzero(selected))
    if not pixel_count:
        return {
            "pixel_count": 0,
            "mean_hue_circular_saturation_weighted_0_to_1": None,
            "mean_saturation": None,
            "mean_value": None,
        }
    hsv = _rgb_hsv(rgb)[selected]
    hue = _circular_hue_mean(hsv[:, 0], hsv[:, 1])
    return {
        "pixel_count": pixel_count,
        "mean_hue_circular_saturation_weighted_0_to_1": hue,
        "mean_saturation": float(np.mean(hsv[:, 1])),
        "mean_value": float(np.mean(hsv[:, 2])),
    }


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def aggregate_rendered_hsv(frames):
    eligible = [frame for frame in frames if frame["liquid_hsv"]["pixel_count"]]
    saturations = [frame["liquid_hsv"]["mean_saturation"] for frame in eligible]
    values = [frame["liquid_hsv"]["mean_value"] for frame in eligible]
    hue_frames = [
        frame
        for frame in eligible
        if frame["liquid_hsv"][
            "mean_hue_circular_saturation_weighted_0_to_1"
        ] is not None
    ]
    hues = [
        frame["liquid_hsv"][
            "mean_hue_circular_saturation_weighted_0_to_1"
        ]
        for frame in hue_frames
    ]
    hue_weights = [frame["liquid_hsv"]["mean_saturation"] for frame in hue_frames]
    saturated = sum(SATURATED_RANGE[0] <= value <= SATURATED_RANGE[1] for value in saturations)
    darkest = sum(DARKEST_TAIL_RANGE[0] <= value <= DARKEST_TAIL_RANGE[1] for value in values)
    count = len(eligible)
    return {
        "eligible_frame_count": count,
        "mean_hue_circular_saturation_weighted_0_to_1": _circular_hue_mean(
            hues, hue_weights
        ),
        "saturation": _distribution(saturations),
        "value": _distribution(values),
        "saturation_0_30_to_0_85": {
            "range_inclusive": list(SATURATED_RANGE),
            "count": saturated,
            "fraction_of_eligible": float(saturated / count) if count else None,
        },
        "value_0_05_to_0_09": {
            "range_inclusive": list(DARKEST_TAIL_RANGE),
            "count": darkest,
            "fraction_of_eligible": float(darkest / count) if count else None,
            "below_range_count": sum(value < DARKEST_TAIL_RANGE[0] for value in values),
            "above_range_count": sum(value > DARKEST_TAIL_RANGE[1] for value in values),
        },
    }


def _v4_visual_contract(frames):
    rendered = aggregate_rendered_hsv(frames)
    values = [
        frame["liquid_hsv"]["mean_value"]
        for frame in frames
        if frame["liquid_hsv"]["pixel_count"]
    ]
    low, high = V4_LOOK_VALUE_REFERENCE_RANGE
    in_reference = sum(low <= value <= high for value in values)
    count = len(values)
    return {
        "name": "accepted-pilot-v4-look",
        "review_status": "manual_review_required",
        "automated_pass_fail_gate_applied": False,
        "reference": {
            "rendered_liquid_value_approximate_range": [low, high],
            "basis": (
                "2026-11-01 ~13:14 CST review: use the exact accepted pilot-v4 "
                "look and expect rendered liquid V about 0.09-0.19"
            ),
            "manual_montage_traits": [
                "wrapped printed labels",
                "black holder",
                "light near-neutral backdrop",
                "grey or dark surface ovals on dark liquids",
                "v4-like liquid surface cues with the accepted v4 glass limitations",
            ],
        },
        "observed": {
            "eligible_frame_count": count,
            "rendered_liquid_saturation": rendered["saturation"],
            "rendered_liquid_value": rendered["value"],
            "value_in_approximate_reference_range": {
                "range_inclusive": [low, high],
                "count": in_reference,
                "fraction_of_eligible": (
                    float(in_reference / count) if count else None
                ),
                "below_range_count": sum(value < low for value in values),
                "above_range_count": sum(value > high for value in values),
            },
        },
        "v5_gates": {
            "saturation_0_30_to_0_85": {
                "applied": False,
                "reason": "colour is added later as mask-guided training augmentation",
            },
            "value_0_05_to_0_09": {
                "applied": False,
                "reason": "the darkest tail is added later during training",
            },
        },
    }


def _breakdowns(frames):
    result = {}
    for field in ("hue_family", "saturation_band", "darkness_band"):
        groups = defaultdict(list)
        for frame in frames:
            groups[frame["material"][field]].append(frame)
        result[field] = {
            name: {
                "frame_count": len(group),
                "rendered_liquid_hsv": aggregate_rendered_hsv(group),
            }
            for name, group in sorted(groups.items())
        }
    return result


def _validate_source_evidence(summary):
    before = summary.get("source_station_tree_hashes_before")
    after = summary.get("source_station_tree_hashes_after")
    if not isinstance(before, dict) or not before:
        raise ValueError("render summary lacks source station hashes before rendering")
    if not isinstance(after, dict) or not after:
        raise ValueError("render summary lacks source station hashes after rendering")
    normalized_before = {
        str(path): _sha(value, f"source station hash before for {path}")
        for path, value in before.items()
    }
    normalized_after = {
        str(path): _sha(value, f"source station hash after for {path}")
        for path, value in after.items()
    }
    if normalized_before != normalized_after:
        raise ValueError("render summary source station hashes changed")
    camera_before = _sha(
        summary.get("source_camera_json_sha256_before"),
        "source camera hash before rendering",
    )
    camera_after = _sha(
        summary.get("source_camera_json_sha256_after"),
        "source camera hash after rendering",
    )
    if camera_before != camera_after:
        raise ValueError("render summary source camera hash changed")
    return {
        "source_station_tree_hashes_before": normalized_before,
        "source_station_tree_hashes_after": normalized_after,
        "source_camera_json_sha256_before": camera_before,
        "source_camera_json_sha256_after": camera_after,
    }


def _validate_exact_glass_params(value):
    if not isinstance(value, dict):
        raise ValueError("render summary lacks the clear active-vial glass override")
    normalized = {}
    for name, expected in EXPECTED_GLASS_PARAMS.items():
        actual = value.get(name)
        if isinstance(expected, list):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                raise ValueError(f"glass {name} must contain {len(expected)} channels")
            channels = [
                _finite(channel, f"glass {name} channel") for channel in actual
            ]
            if name == "specular_rgb" and max(channels) > 0.06:
                raise ValueError(
                    "glass specular_rgb exceeds the dielectric F0 guard"
                )
            if any(
                not math.isclose(channel, target, rel_tol=0.0, abs_tol=1e-12)
                for channel, target in zip(channels, expected)
            ):
                raise ValueError(f"glass {name} does not match the real-look contract")
            normalized[name] = channels
        else:
            number = _finite(actual, f"glass {name}")
            if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"glass {name} does not match the real-look contract")
            normalized[name] = int(number) if name == "use_specular_workflow" else number
    return normalized


def _normalized_usd_prim_path(value, description):
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{description} must be an absolute USD prim path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{description} is not a normalized USD prim path")
    if value == "/":
        raise ValueError(f"{description} must not be the pseudo-root")
    return value


def _validate_glass_prim_paths(summary):
    active = _normalized_usd_prim_path(
        summary.get("active_vial_prim"), "active vial prim"
    )
    paths = summary.get("active_vial_glass_material_prim_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("render summary lacks active-vial glass material prim paths")
    normalized = [
        _normalized_usd_prim_path(value, "active-vial glass material prim path")
        for value in paths
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("active-vial glass material prim paths must be unique")
    prefix = active.rstrip("/") + "/"
    for path in normalized:
        if not path.startswith(prefix):
            raise ValueError(
                "active-vial glass material prim path is not a strict descendant "
                "of active_vial_prim"
            )
    return active, normalized


def _validate_v4_stage_glass(summary):
    active = _normalized_usd_prim_path(
        summary.get("active_vial_prim"), "active vial prim"
    )
    override_fields = (
        "active_vial_glass_material_params",
        "active_vial_glass_material_prim_paths",
    )
    present = [name for name in override_fields if name in summary]
    if present:
        raise ValueError(
            "v4-look render summary unexpectedly authors an active-vial glass "
            f"override: {', '.join(present)}"
        )
    return active, {
        "mode": "stage_v4_unchanged",
        "renderer_authored_override": False,
        "f0_gate": "not_applicable",
        "reason": (
            "the exact archived pilot-v4 renderer leaves the source-stage glass "
            "unchanged and does not author the newer explicit glass/F0 contract"
        ),
    }


def _validate_render_summary(
    summary,
    summary_path,
    manifest_path,
    chunk_dir,
    count,
    *,
    analysis_mode=DEFAULT_ANALYSIS_MODE,
):
    contract = _analysis_contract(analysis_mode)
    required_truths = (
        "passed",
        "source_unchanged",
        "integrity_passed",
        "real_look_requested",
    )
    for name in required_truths:
        if summary.get(name) is not True:
            raise ValueError(f"render summary {name} is not true")
    if summary.get("schema_version") != 1:
        raise ValueError("render summary schema version is not 1")
    if summary.get("frame_count") != count:
        raise ValueError("render summary frame count does not match the train chunk")
    if summary.get("renderer") != EXPECTED_RENDERER:
        raise ValueError("render summary did not use PathTracing")
    if summary.get("subframes") != EXPECTED_SUBFRAMES:
        raise ValueError("render summary did not use 32 subframes")
    if summary.get("resolution") != EXPECTED_RESOLUTION:
        raise ValueError("render summary resolution is not 640x480")
    if summary.get("label_ids") != LABEL_IDS:
        raise ValueError("render summary label IDs do not match the capture contract")
    manifest_hash = file_sha256(manifest_path)
    if summary.get("manifest_sha256") != manifest_hash:
        raise ValueError("render summary manifest hash mismatch")
    hashes = {
        name: _sha(summary.get(name), f"render summary {name}")
        for name in (
            "renderer_script_sha256",
            "variation_module_sha256",
            "specs_sha256",
            "session_layer_sha256",
        )
    }
    session_layer_path = Path(chunk_dir) / "session-layer.usda"
    if not session_layer_path.is_file():
        raise ValueError("train chunk lacks the exported session-layer.usda")
    if file_sha256(session_layer_path) != hashes["session_layer_sha256"]:
        raise ValueError("render summary session-layer hash mismatch")
    source_evidence = _validate_source_evidence(summary)
    if contract["mode"] == V4_LOOK_ANALYSIS_MODE:
        pinned = contract["pinned_hashes"]
        for name in ("renderer_script_sha256", "variation_module_sha256"):
            if hashes[name] != pinned[name]:
                raise ValueError(
                    f"render summary {name} does not match the pinned v4-look input"
                )
        camera_match_hash = _sha(
            summary.get("camera_match_sha256"),
            "render summary camera_match_sha256",
        )
        if camera_match_hash != pinned["camera_match_sha256"]:
            raise ValueError(
                "render summary camera_match_sha256 does not match the pinned "
                "v4-look input"
            )
        active_vial_prim, glass_contract = _validate_v4_stage_glass(summary)
        glass_evidence = {
            "camera_match_sha256": camera_match_hash,
            "active_vial_prim": active_vial_prim,
            "glass_contract": glass_contract,
        }
    else:
        glass = _validate_exact_glass_params(
            summary.get("active_vial_glass_material_params")
        )
        active_vial_prim, paths = _validate_glass_prim_paths(summary)
        glass_evidence = {
            "active_vial_prim": active_vial_prim,
            "active_vial_glass_material_params": glass,
            "active_vial_glass_material_prim_paths": paths,
        }
    return {
        "summary_sha256": file_sha256(summary_path),
        "manifest_sha256": manifest_hash,
        **hashes,
        **source_evidence,
        "source_unchanged": True,
        "integrity_passed": True,
        "real_look_requested": True,
        **glass_evidence,
    }


def _validate_bbox(value, width, height):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("vial bbox must contain four integers")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError("vial bbox must contain four integers")
    x0, y0, x1, y1 = value
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("vial bbox lies outside the RGB image")
    return value


def _preflight_train_records(chunk_dir, records, expected_seeds):
    """Reject non-train identities and unsafe/reused outputs before any decode."""
    for index, (record, expected_seed) in enumerate(zip(records, expected_seeds)):
        if record.get("role") != "train":
            raise ValueError("first-chunk analysis refuses non-train records")
        if record.get("seed") != expected_seed:
            raise ValueError("train manifest seeds are not the expected contiguous range")
        if record.get("frame_index") != index:
            raise ValueError("train manifest frame_index is not contiguous")
        if record.get("variation_profile") != EXPECTED_PROFILE:
            raise ValueError("train record does not use real-look-v2")
    used_paths = {}
    for index, record in enumerate(records):
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError("train record outputs must be an object")
        for name in ("rgb", "mask"):
            path = _safe_output(chunk_dir, outputs.get(name), name.upper())
            previous = used_paths.get(path)
            if previous is not None:
                raise ValueError(
                    f"output file is reused by {previous} and record {index} {name}"
                )
            used_paths[path] = f"record {index} {name}"


def _mask_counts(mask):
    identifiers, counts = np.unique(mask, return_counts=True)
    checker_counts = {
        str(int(identifier)): int(count)
        for identifier, count in zip(identifiers, counts)
    }
    named = {
        name: int(np.count_nonzero(mask == identifier))
        for name, identifier in LABEL_IDS.items()
    }
    vial_count = int(mask.size - named["ignore"])
    fractions = {
        name: (float(named[name] / vial_count) if vial_count else None)
        for name in ("empty", "liquid", "foam", "blocked")
    }
    return checker_counts, named, vial_count, fractions


def _decode_frame(
    chunk_dir,
    record,
    expected_seed,
    expected_index,
    *,
    analysis_mode=DEFAULT_ANALYSIS_MODE,
):
    analysis_contract = _analysis_contract(analysis_mode)
    if record.get("role") != "train":
        raise ValueError("first-chunk analysis refuses non-train records")
    if record.get("seed") != expected_seed:
        raise ValueError("train manifest seeds are not the expected contiguous range")
    if record.get("frame_index") != expected_index:
        raise ValueError("train manifest frame_index is not contiguous")
    if record.get("variation_profile") != EXPECTED_PROFILE:
        raise ValueError("train record does not use real-look-v2")
    outputs = record.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("train record outputs must be an object")
    rgb_path = _safe_output(chunk_dir, outputs.get("rgb"), "RGB")
    mask_path = _safe_output(chunk_dir, outputs.get("mask"), "mask")
    with Image.open(rgb_path) as source:
        if source.mode != "RGB":
            raise ValueError("RGB output must use RGB mode")
        rgb = np.asarray(source)
    with Image.open(mask_path) as source:
        mask = np.asarray(source)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("RGB output must be 640x480 uint8 RGB")
    if mask.shape != (480, 640) or mask.dtype != np.uint8:
        raise ValueError("mask output must be 640x480 uint8")
    unexpected = set(int(value) for value in np.unique(mask)) - ALLOWED_MASK_IDS
    if unexpected:
        raise ValueError(f"mask contains unexpected class identifiers: {sorted(unexpected)}")
    checker_counts, class_counts, vial_count, class_fractions = _mask_counts(mask)
    liquid = record.get("liquid")
    if not isinstance(liquid, dict):
        raise ValueError("train record liquid must be an object")
    material_hsv = liquid.get("hsv")
    if not isinstance(material_hsv, list) or len(material_hsv) != 3:
        raise ValueError("material HSV must contain three channels")
    material_hsv = [_finite(value, "material HSV channel") for value in material_hsv]
    for value in material_hsv:
        if not 0.0 <= value <= 1.0:
            raise ValueError("material HSV channels must be in [0, 1]")
    saturation_band = liquid.get("saturation_band")
    darkness_band = liquid.get("darkness_band")
    family = liquid.get("hue_family")
    if analysis_contract["mode"] == V4_LOOK_ANALYSIS_MODE:
        if "saturation_band" in liquid:
            raise ValueError(
                "exact v4-look material must not author a saturation band"
            )
        saturation_band = "not_authored_v4"
        if darkness_band not in {"very_dark", "dark"}:
            raise ValueError("unsupported v4-look material darkness band")
    else:
        if saturation_band not in {"saturated", "subdued"}:
            raise ValueError("unsupported material saturation band")
        if darkness_band not in {"extreme_dark", "very_dark", "dark"}:
            raise ValueError("unsupported material darkness band")
    if not isinstance(family, str) or not family:
        raise ValueError("material hue family must be a non-empty string")
    label_stats = record.get("label_stats")
    bbox = _validate_bbox(
        label_stats.get("vial_bbox_xyxy") if isinstance(label_stats, dict) else None,
        640,
        480,
    )
    metrics = liquid_hsv_metrics(rgb, mask == LIQUID_ID)
    try:
        level_ml = float(record["level_ml"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("train record level_ml must be numeric") from error
    if not math.isfinite(level_ml) or not 0.0 <= level_ml <= 85.0:
        raise ValueError("train record level_ml is outside [0, 85]")
    return {
        "frame_index": expected_index,
        "seed": expected_seed,
        "role": "train",
        "level_ml": level_ml,
        "material": {
            "hsv": material_hsv,
            "hue_family": family,
            "saturation_band": saturation_band,
            "darkness_band": darkness_band,
        },
        "liquid_hsv": metrics,
        "class_pixel_counts": class_counts,
        "vial_pixel_count": vial_count,
        "class_fractions_of_vial": class_fractions,
        "vial_bbox_xyxy": bbox,
        "outputs": {
            "rgb": {"path": outputs["rgb"]["path"], "sha256": outputs["rgb"]["sha256"]},
            "mask": {"path": outputs["mask"]["path"], "sha256": outputs["mask"]["sha256"]},
        },
        "_rgb_path": rgb_path,
        "_checker_label_counts": checker_counts,
    }


def _select_montage_frames_with_rationale(frames, count=20):
    eligible = [frame for frame in frames if frame["liquid_hsv"]["pixel_count"]]
    selected = []
    reasons = {}

    def add(frame, reason):
        seed = frame["seed"]
        if seed in reasons:
            if reason not in reasons[seed]:
                reasons[seed].append(reason)
            return
        if len(selected) < count:
            selected.append(frame)
            reasons[seed] = [reason]

    positive_zero_liquid = sorted(
        (
            frame
            for frame in frames
            if frame["level_ml"] > 0.0
            and frame["liquid_hsv"]["pixel_count"] == 0
        ),
        key=lambda frame: (frame["level_ml"], frame["seed"]),
    )
    for frame in positive_zero_liquid[:2]:
        add(frame, "positive_level_zero_liquid_glass_focus")

    positive_levels = sorted(
        (frame for frame in frames if frame["level_ml"] > 0.0),
        key=lambda frame: (frame["level_ml"], frame["seed"]),
    )
    for frame in positive_levels[:2]:
        add(frame, "lowest_positive_level_glass_focus")

    for frame in sorted(
        frames,
        key=lambda item: (
            -item["class_fractions_of_vial"]["empty"],
            item["level_ml"],
            item["seed"],
        ),
    )[:2]:
        add(frame, "highest_visible_empty_fraction_glass_focus")

    for band in ("saturated", "subdued"):
        for family in sorted({frame["material"]["hue_family"] for frame in eligible}):
            candidates = [
                frame for frame in eligible
                if frame["material"]["saturation_band"] == band
                and frame["material"]["hue_family"] == family
            ]
            if candidates:
                add(
                    min(candidates, key=lambda frame: frame["seed"]),
                    f"material_family_{family}_{band}",
                )
    for darkness in ("extreme_dark", "very_dark", "dark"):
        candidates = [
            frame
            for frame in eligible
            if frame["material"]["darkness_band"] == darkness
        ]
        if candidates:
            add(
                min(candidates, key=lambda frame: frame["seed"]),
                f"material_darkness_{darkness}",
            )
    for frame in sorted(
        eligible,
        key=lambda item: (item["liquid_hsv"]["mean_value"], item["seed"]),
    ):
        add(frame, "darkest_rendered_liquid")
    for frame in sorted(
        eligible,
        key=lambda item: (-item["liquid_hsv"]["mean_saturation"], item["seed"]),
    ):
        add(frame, "highest_rendered_saturation")
    for frame in sorted(frames, key=lambda item: item["seed"]):
        add(frame, "deterministic_seed_fill")
    if len(selected) != min(count, len(frames)):
        raise ValueError("could not select the requested montage frames")
    metadata = []
    for frame in selected:
        metrics = frame["liquid_hsv"]
        metadata.append(
            {
                "seed": frame["seed"],
                "level_ml": frame["level_ml"],
                "reasons": reasons[frame["seed"]],
                "liquid_pixel_count": metrics["pixel_count"],
                "empty_pixel_count": frame["class_pixel_counts"]["empty"],
                "vial_pixel_count": frame["vial_pixel_count"],
                "liquid_fraction_of_vial": frame["class_fractions_of_vial"][
                    "liquid"
                ],
                "empty_fraction_of_vial": frame["class_fractions_of_vial"][
                    "empty"
                ],
                "material": frame["material"],
                "rendered_liquid_hsv": metrics,
            }
        )
    return selected, metadata


def select_montage_frames(frames, count=20):
    """Return deterministic train-only montage frames; metadata is written separately."""
    selected, _ = _select_montage_frames_with_rationale(frames, count=count)
    return selected


def write_montage(path, frames):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite montage: {path}")
    if any(frame.get("role") != "train" for frame in frames):
        raise ValueError("synthetic montage refuses non-train frames")
    selected, selections = _select_montage_frames_with_rationale(frames)
    columns = 5
    rows = int(math.ceil(len(selected) / columns))
    tile_width, image_height, caption_height = 150, 300, 78
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (image_height + caption_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(selected):
        with Image.open(frame["_rgb_path"]) as source:
            rgb = source.convert("RGB")
            x0, y0, x1, y1 = frame["vial_bbox_xyxy"]
            x0 = max(0, x0 - 24)
            x1 = min(rgb.width, x1 + 24)
            y0 = max(0, y0 - 4)
            y1 = min(rgb.height, y1 + 4)
            crop = rgb.crop((x0, y0, x1, y1))
            crop.thumbnail((tile_width - 8, image_height - 8), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        tile_x, tile_y = col * tile_width, row * (image_height + caption_height)
        image_x = tile_x + (tile_width - crop.width) // 2
        image_y = tile_y + (image_height - crop.height) // 2
        canvas.paste(crop, (image_x, image_y))
        metrics = frame["liquid_hsv"]
        hue = metrics["mean_hue_circular_saturation_weighted_0_to_1"]
        hue_text = "n/a" if hue is None else f"{hue:.3f}"
        saturation = metrics["mean_saturation"]
        value = metrics["mean_value"]
        saturation_text = "n/a" if saturation is None else f"{saturation:.3f}"
        value_text = "n/a" if value is None else f"{value:.3f}"
        caption = (
            f"seed {frame['seed']} {frame['material']['hue_family']}\n"
            f"{frame['material']['saturation_band']} / {frame['material']['darkness_band']}\n"
            f"level {frame['level_ml']:.3f} mL  liquid {metrics['pixel_count']}\n"
            f"H {hue_text}  S {saturation_text}  V {value_text}"
        )
        draw.multiline_text((tile_x + 3, tile_y + image_height + 2), caption, fill="black", spacing=2)
        border = "#14833b" if frame["material"]["saturation_band"] == "saturated" else "#555555"
        draw.rectangle(
            (tile_x, tile_y, tile_x + tile_width - 1, tile_y + image_height + caption_height - 1),
            outline=border,
            width=2,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "selected_seeds": [frame["seed"] for frame in selected],
        "selections": selections,
        "columns": columns,
        "rows": rows,
        "synthetic_only": True,
        "contains_real_frames": False,
        "contains_test_frames": False,
    }


def summarize_train_chunk(
    chunk_dir,
    expected_seed_start,
    expected_seed_end,
    *,
    expected_role,
    montage_out,
    expected_specs_sha256,
    expected_renderer_sha256,
    expected_variation_sha256,
    analysis_mode=DEFAULT_ANALYSIS_MODE,
):
    analysis_contract = _analysis_contract(analysis_mode)
    if expected_role != "train":
        raise ValueError("first-chunk analysis requires expected_role='train'")
    normalized_expected_hashes = {
        "specs_sha256": _sha(
            expected_specs_sha256, "expected specs_sha256"
        ),
        "renderer_script_sha256": _sha(
            expected_renderer_sha256, "expected renderer_script_sha256"
        ),
        "variation_module_sha256": _sha(
            expected_variation_sha256, "expected variation_module_sha256"
        ),
    }
    if analysis_contract["mode"] == V4_LOOK_ANALYSIS_MODE:
        for name in ("renderer_script_sha256", "variation_module_sha256"):
            if (
                normalized_expected_hashes[name]
                != analysis_contract["pinned_hashes"][name]
            ):
                raise ValueError(
                    f"expected {name} does not match the pinned v4-look input"
                )
    montage_out = Path(montage_out).expanduser().resolve()
    if montage_out.exists():
        raise FileExistsError(f"refusing to overwrite montage: {montage_out}")
    chunk_dir = Path(chunk_dir).expanduser().resolve()
    start = int(expected_seed_start)
    end = int(expected_seed_end)
    count = end - start + 1
    expected_count = analysis_contract["expected_frame_count"]
    if count != expected_count:
        raise ValueError(
            f"{analysis_contract['mode']} first train chunk must contain exactly "
            f"{expected_count} expected seeds"
        )
    manifest_path = chunk_dir / "manifest.jsonl"
    summary_path = chunk_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise ValueError("train chunk needs manifest.jsonl and summary.json")
    records = _read_manifest(manifest_path)
    if len(records) != count:
        raise ValueError(
            f"train manifest does not contain exactly {expected_count} records"
        )
    render_summary = _read_json(summary_path, "render summary")
    provenance = _validate_render_summary(
        render_summary,
        summary_path,
        manifest_path,
        chunk_dir,
        count,
        analysis_mode=analysis_contract["mode"],
    )
    for name, expected in normalized_expected_hashes.items():
        if provenance[name] != expected:
            raise ValueError(f"render summary {name} does not match the intended input")
    expected_seeds = tuple(range(start, end + 1))
    _preflight_train_records(chunk_dir, records, expected_seeds)
    independent_checker = _validate_with_independent_checker(
        manifest_path,
        expected_seeds,
        expected_role=expected_role,
        expected_checker_sha256=(
            analysis_contract["pinned_hashes"]["checker_sha256"]
            if analysis_contract["pinned_hashes"] is not None
            else None
        ),
    )
    frames = [
        _decode_frame(
            chunk_dir,
            record,
            seed,
            index,
            analysis_mode=analysis_contract["mode"],
        )
        for index, (record, seed) in enumerate(zip(records, expected_seeds))
    ]
    checker_records = independent_checker["report"]["records"]
    for frame, checker_record in zip(frames, checker_records):
        if frame["_checker_label_counts"] != checker_record["label_counts"]:
            raise ValueError(
                "decoded label counts do not match independent checker evidence "
                f"for seed {frame['seed']}"
            )
    montage = write_montage(montage_out, frames)
    clean_frames = [
        {key: value for key, value in frame.items() if not key.startswith("_")}
        for frame in frames
    ]
    zero_liquid_positive = [
        {
            "seed": frame["seed"],
            "level_ml": frame["level_ml"],
            "material": frame["material"],
            "class_pixel_counts": frame["class_pixel_counts"],
            "vial_pixel_count": frame["vial_pixel_count"],
            "class_fractions_of_vial": frame["class_fractions_of_vial"],
            "explanation": (
                "positive target level with no visible liquid-class pixels; retained "
                "for explicit glass/occlusion review and not treated as an analyzer failure"
            ),
        }
        for frame in clean_frames
        if frame["level_ml"] > 0.0 and frame["liquid_hsv"]["pixel_count"] == 0
    ]
    report = {
        "schema_version": 1,
        "purpose": "Step 8 Revision 2 first train-chunk look review",
        "evidence_label": "measured synthetic render",
        "scope": {
            "role": "train",
            "seed_start": start,
            "seed_end": end,
            "frame_count": count,
            "sealed_test_frames_opened": 0,
            "real_frames_opened": 0,
        },
        "provenance": {
            "manifest_sha256": file_sha256(manifest_path),
            "expected_external_hashes": normalized_expected_hashes,
            **provenance,
        },
        "independent_capture_checker": independent_checker,
        "rendered_liquid_hsv": aggregate_rendered_hsv(clean_frames),
        "breakdowns": _breakdowns(clean_frames),
        "positive_level_zero_liquid_frames": zero_liquid_positive,
        "frames": clean_frames,
        "montage": montage,
    }
    if analysis_contract["mode"] == V4_LOOK_ANALYSIS_MODE:
        report.update(
            {
                "analysis_mode": V4_LOOK_ANALYSIS_MODE,
                "purpose": "Step 8 Revision 2 first v4-look train-chunk review",
                "v4_look_pinned_hashes": dict(V4_LOOK_PINNED_HASHES),
                "v4_visual_contract": _v4_visual_contract(clean_frames),
            }
        )
    return report


def write_json_no_overwrite(path, value):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

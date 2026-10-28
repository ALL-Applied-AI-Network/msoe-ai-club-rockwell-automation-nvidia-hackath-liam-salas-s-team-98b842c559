"""Deterministic temporal-RGB liquid-mask baseline.

This module deliberately separates predictor inputs (two RGB frames) from semantic
training targets. The validation runner never opens evaluation pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError


_RGB_THRESHOLDS = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
_LAB_THRESHOLDS = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
_MARGINS = (0, 4)
_OUTPUT_RULES = ("largest_component", "height_fill_prior")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PREDICTOR_FIELDS = {
    "image_id",
    "run_id",
    "split_role",
    "stage_rgb_path",
    "stage_rgb_sha256",
    "reference_rgb_path",
    "reference_rgb_sha256",
}


@dataclass(frozen=True)
class TemporalMaskConfig:
    """One member of the pre-registered deterministic candidate grid."""

    feature: str
    threshold: float
    support_margin_px: int
    output_rule: str
    minimum_changed_pixels_per_row: int = 4

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


@dataclass(frozen=True)
class TrainGeometry:
    """Train-only union support used by the fixed-camera baseline."""

    union_support: np.ndarray
    bounding_box_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class _Example:
    predictor_record: dict[str, object]
    stage_rgb: np.ndarray
    reference_rgb: np.ndarray
    target_mask: np.ndarray


def candidate_configs() -> list[TemporalMaskConfig]:
    """Return exactly the 56 pre-registered configurations."""

    configs: list[TemporalMaskConfig] = []
    for feature, thresholds in (
        ("max_abs_srgb", _RGB_THRESHOLDS),
        ("lab_delta_e76", _LAB_THRESHOLDS),
    ):
        for threshold in thresholds:
            for margin in _MARGINS:
                for output_rule in _OUTPUT_RULES:
                    configs.append(
                        TemporalMaskConfig(
                            feature=feature,
                            threshold=threshold,
                            support_margin_px=margin,
                            output_rule=output_rule,
                        )
                    )
    return configs


def _bool_mask(value: object, role: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{role} must be a two-dimensional mask")
    return array.astype(bool, copy=False)


def fit_train_geometry(masks: Sequence[object]) -> TrainGeometry:
    """Fit only a union support and bounding box from train masks."""

    if not isinstance(masks, Sequence) or not masks:
        raise ValueError("train masks must be a nonempty sequence")
    checked = [_bool_mask(mask, "train mask") for mask in masks]
    shape = checked[0].shape
    if any(mask.shape != shape for mask in checked):
        raise ValueError("train mask shapes differ")
    union = np.logical_or.reduce(checked)
    rows, columns = np.nonzero(union)
    if rows.size == 0:
        raise ValueError("train mask union is empty")
    return TrainGeometry(
        union_support=union.copy(),
        bounding_box_xyxy=(
            int(columns.min()),
            int(rows.min()),
            int(columns.max()),
            int(rows.max()),
        ),
    )


def _validate_rgb_pair(
    stage_rgb: object, reference_rgb: object
) -> tuple[np.ndarray, np.ndarray]:
    stage = np.asarray(stage_rgb)
    reference = np.asarray(reference_rgb)
    if (
        stage.dtype != np.uint8
        or reference.dtype != np.uint8
        or stage.ndim != 3
        or stage.shape[-1] != 3
        or reference.shape != stage.shape
    ):
        raise ValueError("stage and reference must be aligned uint8 RGB arrays")
    return stage, reference


def _difference_feature(
    stage_rgb: np.ndarray, reference_rgb: np.ndarray, feature: str
) -> np.ndarray:
    if feature == "max_abs_srgb":
        return np.max(
            np.abs(stage_rgb.astype(np.int16) - reference_rgb.astype(np.int16)),
            axis=2,
        ).astype(np.float64)
    if feature == "lab_delta_e76":
        from skimage.color import rgb2lab

        stage_lab = rgb2lab(stage_rgb.astype(np.float64) / 255.0)
        reference_lab = rgb2lab(reference_rgb.astype(np.float64) / 255.0)
        return np.linalg.norm(stage_lab - reference_lab, axis=2)
    raise ValueError("temporal feature is invalid")


def predict_mask(
    stage_rgb: object,
    reference_rgb: object,
    geometry: TrainGeometry,
    config: TemporalMaskConfig,
) -> np.ndarray:
    """Predict one mask from RGB pixels plus frozen train geometry."""

    if not isinstance(geometry, TrainGeometry):
        raise TypeError("geometry must be TrainGeometry")
    if not isinstance(config, TemporalMaskConfig):
        raise TypeError("config must be TemporalMaskConfig")
    if config.support_margin_px not in _MARGINS:
        raise ValueError("support margin is outside the frozen grid")
    if config.output_rule not in _OUTPUT_RULES:
        raise ValueError("output rule is outside the frozen grid")
    if config.minimum_changed_pixels_per_row != 4:
        raise ValueError("row support differs from the frozen grid")
    if not math.isfinite(config.threshold) or config.threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    stage, reference = _validate_rgb_pair(stage_rgb, reference_rgb)
    support = _bool_mask(geometry.union_support, "train support")
    if support.shape != stage.shape[:2]:
        raise ValueError("train support and RGB shapes differ")

    from scipy import ndimage

    expanded = support
    if config.support_margin_px:
        expanded = ndimage.binary_dilation(
            support,
            structure=np.ones((3, 3), dtype=bool),
            iterations=config.support_margin_px,
        )
    feature = _difference_feature(stage, reference, config.feature)
    changed = (feature >= config.threshold) & expanded

    if config.output_rule == "largest_component":
        labels, count = ndimage.label(
            changed, structure=np.ones((3, 3), dtype=bool)
        )
        if count == 0:
            return np.zeros_like(support)
        areas = np.bincount(labels.ravel())[1:]
        selected = int(np.argmax(areas)) + 1
        return labels == selected

    row_counts = np.count_nonzero(changed, axis=1)
    active_rows = np.flatnonzero(
        row_counts >= config.minimum_changed_pixels_per_row
    )
    if active_rows.size == 0:
        return np.zeros_like(support)
    top = int(active_rows[0])
    return support & (np.arange(support.shape[0])[:, None] >= top)


def _height(mask: np.ndarray) -> int:
    rows = np.flatnonzero(np.any(mask, axis=1))
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def score_mask(prediction: object, target: object) -> dict[str, object]:
    """Score a binary prediction with independently checkable pixel counts."""

    predicted = _bool_mask(prediction, "prediction")
    expected = _bool_mask(target, "target")
    if predicted.shape != expected.shape:
        raise ValueError("prediction and target shapes differ")
    intersection = int(np.count_nonzero(predicted & expected))
    prediction_pixels = int(np.count_nonzero(predicted))
    target_pixels = int(np.count_nonzero(expected))
    union = int(np.count_nonzero(predicted | expected))
    iou = intersection / union if union else 1.0
    denominator = prediction_pixels + target_pixels
    dice = (2 * intersection) / denominator if denominator else 1.0
    precision = intersection / prediction_pixels if prediction_pixels else 0.0
    recall = intersection / target_pixels if target_pixels else 0.0
    prediction_height = _height(predicted)
    target_height = _height(expected)
    return {
        "intersection_pixels": intersection,
        "prediction_pixels": prediction_pixels,
        "target_pixels": target_pixels,
        "union_pixels": union,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "prediction_empty": prediction_pixels == 0,
        "prediction_height_px": prediction_height,
        "target_height_px": target_height,
        "absolute_height_error_px": abs(prediction_height - target_height),
    }


def validate_predictor_record(value: object) -> dict[str, object]:
    """Validate the RGB-only routing record passed to predictor code."""

    if not isinstance(value, Mapping) or set(value) != _PREDICTOR_FIELDS:
        raise ValueError("predictor input fields are invalid")
    record = dict(value)
    for field in ("image_id", "run_id"):
        if not isinstance(record[field], str) or not record[field]:
            raise ValueError(f"predictor input {field} is invalid")
    if record["split_role"] not in {"train", "validation", "evaluation"}:
        raise ValueError("predictor input split role is invalid")
    for role in ("stage_rgb", "reference_rgb"):
        path = record[f"{role}_path"]
        digest = record[f"{role}_sha256"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"predictor input {role} path is invalid")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"predictor input {role} hash is invalid")
    if record["stage_rgb_path"] == record["reference_rgb_path"]:
        raise ValueError("predictor RGB inputs must be distinct")
    return record


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_bound_png(path: Path, expected_sha256: str, mode: str) -> np.ndarray:
    data = path.read_bytes()
    if _sha_bytes(data) != expected_sha256:
        raise ValueError(f"pixel hash mismatch: {path}")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != mode:
                raise ValueError(f"pixel encoding invalid: {path}")
            image.load()
            return np.asarray(image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"pixel decode failed: {path}") from exc


def _confined_path(root: Path, relative: object, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{role} path is invalid")
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{role} path escapes merged root") from exc
    return candidate


def _load_jsonl(path: Path) -> tuple[list[dict[str, object]], str]:
    data = path.read_bytes()
    rows: list[dict[str, object]] = []

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"index contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(
            line,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"index contains nonfinite value {token}")
            ),
        )
        if not isinstance(value, dict):
            raise ValueError(f"index row {number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("index is empty")
    return rows, _sha_bytes(data)


def _stage_declarations(
    rows: list[dict[str, object]], root: Path
) -> tuple[dict[str, list[dict[str, object]]], dict[str, set[str]]]:
    declarations = {role: [] for role in ("train", "validation", "evaluation")}
    vial_ids = {role: set() for role in declarations}
    image_ids: set[str] = set()
    for row in rows:
        split = row.get("split_role")
        if split not in declarations:
            raise ValueError("index split role is invalid")
        case_id = row.get("case_id")
        run_id = row.get("run_id")
        batch_id = row.get("batch_id")
        vial_position = row.get("vial_position")
        stages = row.get("stages")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(run_id, str)
            or not run_id
            or not isinstance(batch_id, str)
            or not batch_id
            or isinstance(vial_position, bool)
            or not isinstance(vial_position, int)
            or vial_position not in (1, 2, 3)
            or not isinstance(stages, list)
            or not stages
        ):
            raise ValueError("index case context is invalid")
        vial_key = f"{batch_id}:V{vial_position}"
        vial_ids[str(split)].add(vial_key)
        reference = (
            root
            / "cases"
            / case_id
            / "app-output"
            / "pre-pour-parity"
            / "f2-prefill-time-0001.png"
        )
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise ValueError("index stage is invalid")
            ordinal = stage.get("stage_ordinal")
            tank = stage.get("tank_id")
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal <= 0
                or tank not in {"t100", "t200", "t400"}
            ):
                raise ValueError("index stage identity is invalid")
            image_id = f"{batch_id}-V{vial_position}-dispense-{ordinal}-{tank}"
            if image_id in image_ids:
                raise ValueError("index image ID is duplicated")
            image_ids.add(image_id)
            stage_path = _confined_path(root, stage.get("rgb_path"), "stage RGB")
            mask_path = _confined_path(
                root, stage.get("liquid_mask_path"), "target mask"
            )
            rgb_sha = stage.get("rgb_sha256")
            mask_sha = stage.get("liquid_mask_sha256")
            if not isinstance(rgb_sha, str) or _SHA256.fullmatch(rgb_sha) is None:
                raise ValueError("stage RGB hash is invalid")
            if not isinstance(mask_sha, str) or _SHA256.fullmatch(mask_sha) is None:
                raise ValueError("target mask hash is invalid")
            declarations[str(split)].append(
                {
                    "image_id": image_id,
                    "run_id": run_id,
                    "split_role": split,
                    "stage_rgb_path": os.fspath(stage_path),
                    "stage_rgb_sha256": rgb_sha,
                    "reference_rgb_path": os.fspath(reference.resolve(strict=False)),
                    "target_mask_path": os.fspath(mask_path),
                    "target_mask_sha256": mask_sha,
                }
            )
    return declarations, vial_ids


def _load_examples(declarations: Sequence[Mapping[str, object]]) -> list[_Example]:
    examples: list[_Example] = []
    reference_cache: dict[str, tuple[np.ndarray, str]] = {}
    for raw in declarations:
        stage_path = Path(str(raw["stage_rgb_path"]))
        target_path = Path(str(raw["target_mask_path"]))
        reference_path = Path(str(raw["reference_rgb_path"]))
        stage = _load_bound_png(stage_path, str(raw["stage_rgb_sha256"]), "RGB")
        target_uint8 = _load_bound_png(
            target_path, str(raw["target_mask_sha256"]), "L"
        )
        if not set(int(value) for value in np.unique(target_uint8)).issubset(
            {0, 255}
        ):
            raise ValueError("target mask is not binary")
        reference_key = os.fspath(reference_path)
        cached = reference_cache.get(reference_key)
        if cached is None:
            reference_data = reference_path.read_bytes()
            reference_sha = _sha_bytes(reference_data)
            reference = _load_bound_png(reference_path, reference_sha, "RGB")
            reference_cache[reference_key] = (reference, reference_sha)
        else:
            reference, reference_sha = cached
        if stage.shape != reference.shape or target_uint8.shape != stage.shape[:2]:
            raise ValueError("RGB and target shapes differ")
        predictor = validate_predictor_record(
            {
                "image_id": raw["image_id"],
                "run_id": raw["run_id"],
                "split_role": raw["split_role"],
                "stage_rgb_path": os.fspath(stage_path),
                "stage_rgb_sha256": raw["stage_rgb_sha256"],
                "reference_rgb_path": reference_key,
                "reference_rgb_sha256": reference_sha,
            }
        )
        examples.append(
            _Example(
                predictor_record=predictor,
                stage_rgb=stage,
                reference_rgb=reference,
                target_mask=target_uint8 == 255,
            )
        )
    return examples


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("metric rows are empty")
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in (
            "iou",
            "dice",
            "precision",
            "recall",
            "absolute_height_error_px",
        )
    }
    return {
        "image_count": len(rows),
        "valid_prediction_count": sum(
            not bool(row["prediction_empty"]) for row in rows
        ),
        "empty_prediction_count": sum(
            bool(row["prediction_empty"]) for row in rows
        ),
        "mean_iou": float(np.mean(values["iou"])),
        "median_iou": float(np.median(values["iou"])),
        "p05_iou": float(np.percentile(values["iou"], 5)),
        "mean_dice": float(np.mean(values["dice"])),
        "median_dice": float(np.median(values["dice"])),
        "p05_dice": float(np.percentile(values["dice"], 5)),
        "mean_precision": float(np.mean(values["precision"])),
        "mean_recall": float(np.mean(values["recall"])),
        "mean_absolute_height_error_px": float(
            np.mean(values["absolute_height_error_px"])
        ),
        "max_absolute_height_error_px": int(
            np.max(values["absolute_height_error_px"])
        ),
    }


def _score_config(
    examples: Sequence[_Example], geometry: TrainGeometry, config: TemporalMaskConfig
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        predicted = predict_mask(
            example.stage_rgb, example.reference_rgb, geometry, config
        )
        rows.append(
            {
                "image_id": example.predictor_record["image_id"],
                **score_mask(predicted, example.target_mask),
            }
        )
    return rows, _aggregate(rows)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _repo_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_diff_sha256": None}
    return {"commit": commit, "dirty_diff_sha256": _sha_bytes(diff)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_validation_experiment(
    *,
    index_path: str | Path,
    merged_root: str | Path,
    preregistration_path: str | Path,
    output_dir: str | Path,
    expected_denominators: Mapping[str, int],
) -> dict[str, object]:
    """Tune on validation while leaving evaluation pixel files unopened."""

    output = Path(output_dir)
    if os.path.lexists(output):
        raise FileExistsError(f"experiment output already exists: {output}")
    index = Path(index_path).resolve(strict=True)
    root = Path(merged_root).resolve(strict=True)
    prereg = Path(preregistration_path).resolve(strict=True)
    rows, index_sha = _load_jsonl(index)
    declarations, vials = _stage_declarations(rows, root)
    actual_denominators = {
        "train_vials": len(vials["train"]),
        "train_stage_images": len(declarations["train"]),
        "validation_vials": len(vials["validation"]),
        "validation_stage_images": len(declarations["validation"]),
        "evaluation_vials_declared_not_opened": len(vials["evaluation"]),
        "evaluation_stage_images_declared_not_opened": len(
            declarations["evaluation"]
        ),
    }
    if dict(expected_denominators) != actual_denominators:
        raise ValueError(
            f"experiment denominators drifted: {actual_denominators!r}"
        )

    train = _load_examples(declarations["train"])
    validation = _load_examples(declarations["validation"])
    geometry = fit_train_geometry([example.target_mask for example in train])

    candidates: list[dict[str, object]] = []
    detail_by_sha: dict[str, list[dict[str, object]]] = {}
    for config in candidate_configs():
        validation_rows, validation_metrics = _score_config(
            validation, geometry, config
        )
        config_dict = asdict(config)
        config_sha = _canonical_sha(config_dict)
        candidates.append(
            {
                "config": config_dict,
                "config_sha256": config_sha,
                "validation": validation_metrics,
                "validation_per_image": validation_rows,
            }
        )
        detail_by_sha[config_sha] = validation_rows
    candidates.sort(
        key=lambda row: (
            -float(row["validation"]["mean_dice"]),
            -float(row["validation"]["p05_dice"]),
            float(row["validation"]["mean_absolute_height_error_px"]),
            json.dumps(row["config"], sort_keys=True, separators=(",", ":")),
        )
    )
    selected_record = candidates[0]
    selected = TemporalMaskConfig(**selected_record["config"])
    train_rows, train_metrics = _score_config(train, geometry, selected)
    validation_rows = detail_by_sha[str(selected_record["config_sha256"])]

    majority = np.mean(
        np.stack([example.target_mask for example in train], axis=0), axis=0
    ) >= 0.5
    control_rows = [
        {
            "image_id": example.predictor_record["image_id"],
            **score_mask(majority, example.target_mask),
        }
        for example in validation
    ]
    control_metrics = _aggregate(control_rows)
    selected_metrics = selected_record["validation"]
    gate_checks = {
        "empty_prediction_count_is_zero": selected_metrics[
            "empty_prediction_count"
        ]
        == 0,
        "mean_dice_at_least_0_85": selected_metrics["mean_dice"] >= 0.85,
        "p05_dice_at_least_0_70": selected_metrics["p05_dice"] >= 0.70,
        "mean_height_error_at_most_5_px": selected_metrics[
            "mean_absolute_height_error_px"
        ]
        <= 5.0,
        "beats_location_only_control_mean_dice": selected_metrics["mean_dice"]
        > control_metrics["mean_dice"],
    }
    gate_passed = all(gate_checks.values())

    geometry_buffer = BytesIO()
    Image.fromarray(
        geometry.union_support.astype(np.uint8) * 255, mode="L"
    ).save(geometry_buffer, format="PNG")
    geometry_payload = geometry_buffer.getvalue()
    geometry_sha = _sha_bytes(geometry_payload)
    selected_config = {
        "schema_version": 1,
        "artifact_type": "temporal_rgb_mask_baseline_config",
        "status": "frozen_after_validation",
        "config": asdict(selected),
        "config_sha256": selected_record["config_sha256"],
        "train_geometry": {
            "artifact": "train-union-mask.png",
            "sha256": geometry_sha,
            "bounding_box_xyxy": list(geometry.bounding_box_xyxy),
            "pixel_count": int(np.count_nonzero(geometry.union_support)),
        },
        "input_fields": ["rgb"],
        "threshold_source": "twin_validation_frozen",
    }
    predictor_inputs = {
        "schema_version": 1,
        "artifact_type": "temporal_rgb_predictor_inputs",
        "input_fields": ["rgb"],
        "records": [example.predictor_record for example in [*train, *validation]],
        "evaluation_declared_not_opened": {
            "vials": actual_denominators[
                "evaluation_vials_declared_not_opened"
            ],
            "stage_images": actual_denominators[
                "evaluation_stage_images_declared_not_opened"
            ],
        },
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "temporal_rgb_mask_validation_result",
        "status": "validation_pass" if gate_passed else "validation_rejected",
        "hypothesis_result": "supported" if gate_passed else "falsified",
        "candidate_count": len(candidates),
        "seed": 0,
        "source": {
            "index_path": os.fspath(index),
            "index_sha256": index_sha,
            "merged_root": os.fspath(root),
            "preregistration_path": os.fspath(prereg),
            "preregistration_sha256": _file_sha(prereg),
        },
        "repository": _repo_state(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": package_version("Pillow"),
            "scipy": package_version("scipy"),
            "scikit_image": package_version("scikit-image"),
            "platform": platform.platform(),
        },
        "invocation": {"argv": list(sys.argv), "cwd": os.fspath(Path.cwd())},
        "denominators": actual_denominators,
        "selected_config": selected_config,
        "train_metrics": train_metrics,
        "validation_metrics": selected_metrics,
        "validation_per_image": validation_rows,
        "location_only_control": {
            "definition": "pixelwise_train_mask_frequency_at_least_0_5",
            "validation_metrics": control_metrics,
            "validation_per_image": control_rows,
        },
        "validation_gate": {"passed": gate_passed, "checks": gate_checks},
        "evaluation_pixels_opened": False,
        "evaluation_status": "not_run_validation_gate_required",
        "claim_boundary": (
            "Synthetic fixed-FillCam validation only; no unseen-geometry, real-camera, "
            "keep/discard, cause, fluid-physics, or sim-to-real claim."
        ),
    }
    json.dumps(result, allow_nan=False)
    output.mkdir(parents=False, exist_ok=False)
    (output / "train-union-mask.png").write_bytes(geometry_payload)
    _write_json(output / "frozen-config.json", selected_config)
    _write_json(output / "predictor-inputs.json", predictor_inputs)
    _write_json(output / "candidate-results.json", candidates)
    _write_json(output / "train-per-image.json", train_rows)
    _write_json(output / "result.json", result)
    return result

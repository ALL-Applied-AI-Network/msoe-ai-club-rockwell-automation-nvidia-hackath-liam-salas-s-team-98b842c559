"""Deterministic supervised fixed-camera liquid masks from one RGB image.

Semantic masks are accepted only by the training/evaluation orchestration.  The
public prediction boundary is deliberately ``predict_mask(rgb, model, config)``.
Models are explicit JSON tree arrays and never pickle executable Python objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
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


_FEATURE_SETS = ("rgb_xy", "rgb_xy_local")
_DEPTHS = (8, 12)
_MIN_LEAVES = (64, 256)
_THRESHOLDS = (0.35, 0.5)
_CLEANUPS = ("largest_component", "largest_component_row_fill")
_SEED = 20261028
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_FIELDS = {
    "schema_version",
    "artifact_type",
    "feature_set",
    "image_shape_hw",
    "train_union_support_rle",
    "trees",
}
_TREE_FIELDS = {
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "positive_probability",
}


@dataclass(frozen=True)
class SpatialMaskConfig:
    """One member of the preregistered 32-candidate grid."""

    feature_set: str
    max_depth: int
    min_samples_leaf: int
    probability_threshold: float
    cleanup: str
    seed: int = _SEED

    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        )


@dataclass(frozen=True)
class MaskExample:
    image_id: str
    rgb_path: Path
    rgb_sha256: str
    rgb: np.ndarray
    target_mask: np.ndarray


@dataclass(frozen=True)
class ExperimentData:
    examples: dict[str, tuple[MaskExample, ...]]
    declarations: dict[str, tuple[dict[str, object], ...]]
    denominators: dict[str, int]
    index_sha256: str


def canonical_json_bytes(value: object) -> bytes:
    """Return the stable on-disk encoding used for model/config hash binding."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidate_configs() -> list[SpatialMaskConfig]:
    """Return exactly the preregistered configurations in canonical loop order."""

    return [
        SpatialMaskConfig(feature, depth, leaf, threshold, cleanup)
        for feature in _FEATURE_SETS
        for depth in _DEPTHS
        for leaf in _MIN_LEAVES
        for threshold in _THRESHOLDS
        for cleanup in _CLEANUPS
    ]


def _validate_config(config: object) -> SpatialMaskConfig:
    if not isinstance(config, SpatialMaskConfig):
        raise TypeError("config must be SpatialMaskConfig")
    if (
        config.feature_set not in _FEATURE_SETS
        or config.max_depth not in _DEPTHS
        or config.min_samples_leaf not in _MIN_LEAVES
        or config.probability_threshold not in _THRESHOLDS
        or config.cleanup not in _CLEANUPS
        or config.seed != _SEED
    ):
        raise ValueError("config is outside the preregistered grid")
    return config


def _validate_rgb(rgb: object) -> np.ndarray:
    array = np.asarray(rgb)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("predictor input must be one uint8 RGB array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("predictor input must be one nonempty uint8 RGB array")
    return array


def _feature_matrix(rgb: np.ndarray, feature_set: str) -> np.ndarray:
    height, width = rgb.shape[:2]
    rows, columns = np.indices((height, width), dtype=np.float32)
    base = [
        rgb.astype(np.float32) / np.float32(255.0),
        ((columns + np.float32(0.5)) / np.float32(width))[..., None],
        ((rows + np.float32(0.5)) / np.float32(height))[..., None],
    ]
    if feature_set == "rgb_xy_local":
        from scipy import ndimage

        local = ndimage.uniform_filter(
            rgb.astype(np.float32), size=(3, 3, 1), mode="nearest"
        )
        base.extend(
            [
                local / np.float32(255.0),
                np.abs(rgb.astype(np.float32) - local) / np.float32(255.0),
            ]
        )
    elif feature_set != "rgb_xy":
        raise ValueError("model feature set is invalid")
    return np.concatenate(base, axis=2).reshape(height * width, -1)


def _decode_support(model: Mapping[str, object]) -> np.ndarray:
    shape = model["image_shape_hw"]
    assert isinstance(shape, list)
    height, width = int(shape[0]), int(shape[1])
    flat = np.zeros(height * width, dtype=bool)
    rle = model["train_union_support_rle"]
    assert isinstance(rle, list)
    for pair in rle:
        assert isinstance(pair, list)
        start, length = int(pair[0]), int(pair[1])
        flat[start : start + length] = True
    return flat.reshape(height, width)


def _encode_support(mask: np.ndarray) -> list[list[int]]:
    flat = np.asarray(mask, dtype=bool).ravel()
    indices = np.flatnonzero(flat)
    if indices.size == 0:
        raise ValueError("train union support is empty")
    runs: list[list[int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current != previous + 1:
            runs.append([start, previous - start + 1])
            start = current
        previous = current
    runs.append([start, previous - start + 1])
    return runs


def validate_model(value: object) -> dict[str, object]:
    """Validate the complete executable-free random-forest representation."""

    if not isinstance(value, Mapping) or set(value) != _MODEL_FIELDS:
        raise ValueError("model fields are invalid")
    model = dict(value)
    if (
        model["schema_version"] != 1
        or model["artifact_type"] != "spatial_rgb_mask_random_forest"
        or model["feature_set"] not in _FEATURE_SETS
    ):
        raise ValueError("model identity is invalid")
    shape = model["image_shape_hw"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)
    ):
        raise ValueError("model image shape is invalid")
    total = int(shape[0]) * int(shape[1])
    rle = model["train_union_support_rle"]
    if not isinstance(rle, list) or not rle:
        raise ValueError("model support is invalid")
    last_end = 0
    for pair in rle:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
        ):
            raise ValueError("model support is invalid")
        start, length = pair
        if start < last_end or length <= 0 or start + length > total:
            raise ValueError("model support is invalid")
        last_end = start + length
    trees = model["trees"]
    if not isinstance(trees, list) or not trees:
        raise ValueError("model trees are invalid")
    feature_count = 5 if model["feature_set"] == "rgb_xy" else 11
    for tree in trees:
        if not isinstance(tree, Mapping) or set(tree) != _TREE_FIELDS:
            raise ValueError("model tree fields are invalid")
        lengths = [len(tree[field]) for field in _TREE_FIELDS]
        if len(set(lengths)) != 1 or lengths[0] == 0:
            raise ValueError("model tree arrays are invalid")
        count = lengths[0]
        for node, (left, right, feature, threshold, probability) in enumerate(
            zip(
                tree["children_left"],
                tree["children_right"],
                tree["feature"],
                tree["threshold"],
                tree["positive_probability"],
            )
        ):
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (left, right, feature)):
                raise ValueError("model tree index is invalid")
            if left == -1 or right == -1:
                if left != -1 or right != -1:
                    raise ValueError("model tree leaf is invalid")
                if feature != -2:
                    raise ValueError("model tree leaf feature is invalid")
            elif left <= node or right <= node or left >= count or right >= count:
                raise ValueError("model tree child is invalid")
            elif feature < 0 or feature >= feature_count:
                raise ValueError("model tree feature is invalid")
            if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
                raise ValueError("model tree threshold is invalid")
            if not isinstance(probability, (int, float)) or not 0 <= float(probability) <= 1:
                raise ValueError("model tree probability is invalid")
    _decode_support(model)
    return json.loads(canonical_json_bytes(model))


def load_model(path: str | Path, expected_sha256: str) -> dict[str, object]:
    """Load canonical JSON only after checking the caller's frozen byte hash."""

    model_path = Path(path)
    data = model_path.read_bytes()
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("expected model hash is invalid")
    if sha256_bytes(data) != expected_sha256:
        raise ValueError("model hash mismatch")
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model JSON is invalid") from error
    model = validate_model(value)
    if data != canonical_json_bytes(model):
        raise ValueError("model bytes are not canonical")
    return model


def _tree_probability(
    features: np.ndarray, tree: Mapping[str, object], max_depth: int
) -> np.ndarray:
    left = np.asarray(tree["children_left"], dtype=np.int32)
    right = np.asarray(tree["children_right"], dtype=np.int32)
    feature = np.asarray(tree["feature"], dtype=np.int32)
    threshold = np.asarray(tree["threshold"], dtype=np.float32)
    probability = np.asarray(tree["positive_probability"], dtype=np.float32)
    nodes = np.zeros(features.shape[0], dtype=np.int32)
    active = np.ones(features.shape[0], dtype=bool)
    for _ in range(max_depth):
        indices = np.flatnonzero(active)
        if indices.size == 0:
            break
        current = nodes[indices]
        branch = left[current] >= 0
        if not np.any(branch):
            active[indices] = False
            break
        leaf_indices = indices[~branch]
        active[leaf_indices] = False
        branch_indices = indices[branch]
        current = nodes[branch_indices]
        goes_left = features[branch_indices, feature[current]] <= threshold[current]
        nodes[branch_indices] = np.where(goes_left, left[current], right[current])
    return probability[nodes]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    objects = ndimage.find_objects(labels)
    ranked: list[tuple[int, int, int, int]] = []
    for label_id, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        area = int(np.count_nonzero(labels[bounds] == label_id))
        ranked.append((-area, int(bounds[0].start), int(bounds[1].start), label_id))
    selected = min(ranked)[3]
    return labels == selected


def predict_mask(
    rgb: object, model: Mapping[str, object], config: SpatialMaskConfig
) -> np.ndarray:
    """Predict one binary mask from exactly one RGB image and frozen artifacts."""

    image = _validate_rgb(rgb)
    frozen = validate_model(model)
    selected = _validate_config(config)
    if frozen["feature_set"] != selected.feature_set:
        raise ValueError("model and config feature sets differ")
    if list(image.shape[:2]) != frozen["image_shape_hw"]:
        raise ValueError("RGB and model image shapes differ")
    features = _feature_matrix(image, selected.feature_set)
    probabilities = np.mean(
        [
            _tree_probability(features, tree, selected.max_depth)
            for tree in frozen["trees"]
        ],
        axis=0,
    ).reshape(image.shape[:2])
    support = _decode_support(frozen)
    component = _largest_component(
        (probabilities >= selected.probability_threshold) & support
    )
    if selected.cleanup == "largest_component":
        return component
    filled = component.copy()
    for row in np.flatnonzero(np.any(component, axis=1)):
        columns = np.flatnonzero(component[row])
        filled[row, columns[0] : columns[-1] + 1] = True
    return filled & support


def claim_evaluation_pass(path: str | Path) -> None:
    """Atomically and durably claim the sole allowed evaluation pass."""

    sentinel = Path(path)
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(sentinel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"evaluation pass already claimed: {sentinel}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("evaluation pass claimed exactly once\n")
        stream.flush()
        os.fsync(stream.fileno())


def _bound_png(path: Path, expected_sha256: str, mode: str) -> np.ndarray:
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise ValueError(f"pixel hash mismatch: {path}")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.mode != mode:
                raise ValueError(f"pixel encoding invalid: {path}")
            image.load()
            return np.asarray(image, dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"pixel decode failed: {path}") from error


def _confined(root: Path, relative: object, role: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{role} path is invalid")
    path = (root / relative).resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{role} path escapes merged root") from error
    return path


def _index_rows(path: Path) -> tuple[list[dict[str, object]], str]:
    data = path.read_bytes()

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate index key: {key}")
            result[key] = value
        return result

    rows = [
        json.loads(line, object_pairs_hook=unique_pairs)
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("index rows are invalid")
    return rows, sha256_bytes(data)


def _declarations(
    rows: Sequence[Mapping[str, object]], root: Path
) -> tuple[dict[str, list[dict[str, object]]], dict[str, set[str]]]:
    declarations = {role: [] for role in ("train", "validation", "evaluation")}
    vials = {role: set() for role in declarations}
    ids: set[str] = set()
    for row in rows:
        role = row.get("split_role")
        if role not in declarations:
            raise ValueError("index split role is invalid")
        batch = row.get("batch_id")
        case = row.get("case_id")
        position = row.get("vial_position")
        stages = row.get("stages")
        if (
            not isinstance(batch, str)
            or not batch
            or not isinstance(case, str)
            or not case
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not isinstance(stages, list)
            or not stages
        ):
            raise ValueError("index case context is invalid")
        vials[str(role)].add(f"{batch}:V{position}")
        for stage in stages:
            if not isinstance(stage, Mapping):
                raise ValueError("index stage is invalid")
            ordinal = stage.get("stage_ordinal")
            tank = stage.get("tank_id")
            image_id = f"{batch}-V{position}-dispense-{ordinal}-{tank}"
            if (
                isinstance(ordinal, bool)
                or not isinstance(ordinal, int)
                or ordinal <= 0
                or tank not in {"t100", "t200", "t400"}
                or image_id in ids
            ):
                raise ValueError("index stage identity is invalid")
            ids.add(image_id)
            rgb_sha = stage.get("rgb_sha256")
            mask_sha = stage.get("liquid_mask_sha256")
            if (
                not isinstance(rgb_sha, str)
                or _SHA256.fullmatch(rgb_sha) is None
                or not isinstance(mask_sha, str)
                or _SHA256.fullmatch(mask_sha) is None
            ):
                raise ValueError("index stage hashes are invalid")
            declarations[str(role)].append(
                {
                    "image_id": image_id,
                    "rgb_path": _confined(root, stage.get("rgb_path"), "RGB"),
                    "rgb_sha256": rgb_sha,
                    "target_mask_path": _confined(
                        root, stage.get("liquid_mask_path"), "target mask"
                    ),
                    "target_mask_sha256": mask_sha,
                }
            )
    return declarations, vials


def load_experiment_data(
    *,
    index_path: str | Path,
    merged_root: str | Path,
    roles: Sequence[str],
    expected_denominators: Mapping[str, int],
) -> ExperimentData:
    """Load only explicitly named split pixels after checking all declarations."""

    if not roles or any(role not in {"train", "validation", "evaluation"} for role in roles):
        raise ValueError("requested experiment roles are invalid")
    index = Path(index_path).resolve(strict=True)
    root = Path(merged_root).resolve(strict=True)
    rows, index_sha = _index_rows(index)
    declarations, vials = _declarations(rows, root)
    denominators = {
        f"{role}_{unit}": len(vials[role] if unit == "vials" else declarations[role])
        for role in declarations
        for unit in ("vials", "stage_images")
    }
    if denominators != dict(expected_denominators):
        raise ValueError(f"experiment denominators drifted: {denominators!r}")
    loaded: dict[str, tuple[MaskExample, ...]] = {}
    shape: tuple[int, int] | None = None
    for role in roles:
        examples: list[MaskExample] = []
        for declaration in declarations[role]:
            rgb = _bound_png(
                declaration["rgb_path"], declaration["rgb_sha256"], "RGB"
            )
            mask_u8 = _bound_png(
                declaration["target_mask_path"],
                declaration["target_mask_sha256"],
                "L",
            )
            if not set(np.unique(mask_u8).tolist()).issubset({0, 255}):
                raise ValueError("target mask is not binary")
            if mask_u8.shape != rgb.shape[:2]:
                raise ValueError("RGB and target mask shapes differ")
            if shape is None:
                shape = rgb.shape[:2]
            elif shape != rgb.shape[:2]:
                raise ValueError("experiment image shapes differ")
            examples.append(
                MaskExample(
                    image_id=str(declaration["image_id"]),
                    rgb_path=declaration["rgb_path"],
                    rgb_sha256=str(declaration["rgb_sha256"]),
                    rgb=rgb,
                    target_mask=mask_u8 == 255,
                )
            )
        loaded[role] = tuple(examples)
    frozen_declarations = {
        role: tuple(
            {
                "image_id": item["image_id"],
                "rgb_path": os.fspath(item["rgb_path"]),
                "rgb_sha256": item["rgb_sha256"],
                "target_mask_path": os.fspath(item["target_mask_path"]),
                "target_mask_sha256": item["target_mask_sha256"],
            }
            for item in declarations[role]
        )
        for role in declarations
    }
    return ExperimentData(loaded, frozen_declarations, denominators, index_sha)


def build_locator_artifact(
    *,
    predictions: Mapping[str, Mapping[str, str]],
    model_sha256: str,
    config_sha256: str,
    code_commit: str,
) -> dict[str, object]:
    """Build the strict schema-v2 locator without any oracle-like fields."""

    return {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": {
            "producer_id": "spatial-rgb-mask-predictor-01a18128",
            "producer_kind": "rgb_only_liquid_segmenter",
            "model_id": "spatial-rgb-random-forest-v1",
            "model_sha256": model_sha256,
            "config_id": "spatial-rgb-mask-config-v1",
            "config_sha256": config_sha256,
            "code_commit": code_commit,
            "threshold_source": "twin_validation_frozen",
            "training_data_origin": "twin",
            "input_fields": ["rgb"],
        },
        "images": {key: dict(predictions[key]) for key in sorted(predictions)},
    }


def _sample_indices(
    examples: Sequence[MaskExample],
) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(_SEED))
    sampled: dict[str, np.ndarray] = {}
    for example in sorted(examples, key=lambda item: item.image_id):
        target = example.target_mask.ravel()
        if target.size <= 65_536:
            indices = np.arange(target.size, dtype=np.int64)
        else:
            positive = np.flatnonzero(target)
            negative = np.flatnonzero(~target)
            positive = rng.choice(
                positive, size=min(8_192, positive.size), replace=False
            )
            negative = rng.choice(
                negative, size=min(8_192, negative.size), replace=False
            )
            indices = np.sort(np.concatenate([positive, negative]))
        sampled[example.image_id] = indices
    return sampled


def _sample_matrix(
    examples: Sequence[MaskExample],
    sampled_indices: Mapping[str, np.ndarray],
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrices = []
    labels = []
    for example in sorted(examples, key=lambda item: item.image_id):
        indices = sampled_indices[example.image_id]
        matrices.append(_feature_matrix(example.rgb, feature_set)[indices])
        labels.append(example.target_mask.ravel()[indices].astype(np.uint8))
    matrix = np.concatenate(matrices, axis=0)
    target = np.concatenate(labels, axis=0)
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("training sample must contain both mask classes")
    return matrix, target


def _export_forest(
    forest: object, *, feature_set: str, support: np.ndarray
) -> dict[str, object]:
    trees = []
    estimators = getattr(forest, "estimators_", None)
    if not isinstance(estimators, list) or not estimators:
        raise ValueError("fitted forest has no estimators")
    classes = np.asarray(getattr(forest, "classes_", []))
    positive_positions = np.flatnonzero(classes == 1)
    if positive_positions.size != 1:
        raise ValueError("fitted forest classes are invalid")
    positive = int(positive_positions[0])
    for estimator in estimators:
        tree = estimator.tree_
        values = np.asarray(tree.value, dtype=np.float64).reshape(tree.node_count, -1)
        totals = np.sum(values, axis=1)
        probabilities = np.divide(
            values[:, positive],
            totals,
            out=np.zeros(tree.node_count, dtype=np.float64),
            where=totals > 0,
        )
        trees.append(
            {
                "children_left": tree.children_left.astype(int).tolist(),
                "children_right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "positive_probability": probabilities.astype(float).tolist(),
            }
        )
    return validate_model(
        {
            "schema_version": 1,
            "artifact_type": "spatial_rgb_mask_random_forest",
            "feature_set": feature_set,
            "image_shape_hw": list(support.shape),
            "train_union_support_rle": _encode_support(support),
            "trees": trees,
        }
    )


def fit_forest_model(
    examples: Sequence[MaskExample],
    *,
    sampled_indices: Mapping[str, np.ndarray],
    feature_set: str,
    min_samples_leaf: int,
) -> dict[str, object]:
    """Fit one max-depth-12 forest for the depth-8/12 traversal candidates."""

    if not examples:
        raise ValueError("training examples are empty")
    if feature_set not in _FEATURE_SETS or min_samples_leaf not in _MIN_LEAVES:
        raise ValueError("forest fit parameters are outside the preregistered grid")
    shapes = {example.target_mask.shape for example in examples}
    if len(shapes) != 1:
        raise ValueError("training mask shapes differ")
    support = np.logical_or.reduce([example.target_mask for example in examples])
    matrix, target = _sample_matrix(
        examples, sampled_indices, feature_set
    )
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(
        n_estimators=5,
        criterion="gini",
        max_depth=12,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=_SEED,
        n_jobs=1,
    )
    classifier.fit(matrix, target)
    return _export_forest(classifier, feature_set=feature_set, support=support)


def score_mask(prediction: object, target: object) -> dict[str, object]:
    predicted = np.asarray(prediction)
    expected = np.asarray(target)
    if predicted.ndim != 2 or expected.ndim != 2 or predicted.shape != expected.shape:
        raise ValueError("prediction and target must be aligned two-dimensional masks")
    predicted = predicted.astype(bool, copy=False)
    expected = expected.astype(bool, copy=False)
    intersection = int(np.count_nonzero(predicted & expected))
    predicted_count = int(np.count_nonzero(predicted))
    target_count = int(np.count_nonzero(expected))
    union = int(np.count_nonzero(predicted | expected))
    denominator = predicted_count + target_count
    predicted_rows = np.flatnonzero(np.any(predicted, axis=1))
    target_rows = np.flatnonzero(np.any(expected, axis=1))
    predicted_height = (
        int(predicted_rows[-1] - predicted_rows[0] + 1) if predicted_rows.size else 0
    )
    target_height = int(target_rows[-1] - target_rows[0] + 1) if target_rows.size else 0
    return {
        "intersection_pixels": intersection,
        "prediction_pixels": predicted_count,
        "target_pixels": target_count,
        "union_pixels": union,
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection) / denominator if denominator else 1.0,
        "precision": intersection / predicted_count if predicted_count else 0.0,
        "recall": intersection / target_count if target_count else 0.0,
        "prediction_empty": predicted_count == 0,
        "prediction_height_px": predicted_height,
        "target_height_px": target_height,
        "absolute_height_error_px": abs(predicted_height - target_height),
    }


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("metric rows are empty")
    arrays = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in ("iou", "dice", "precision", "recall", "absolute_height_error_px")
    }
    return {
        "image_count": len(rows),
        "valid_prediction_count": sum(not bool(row["prediction_empty"]) for row in rows),
        "empty_prediction_count": sum(bool(row["prediction_empty"]) for row in rows),
        "mean_iou": float(np.mean(arrays["iou"])),
        "median_iou": float(np.median(arrays["iou"])),
        "p05_iou": float(np.percentile(arrays["iou"], 5)),
        "mean_dice": float(np.mean(arrays["dice"])),
        "median_dice": float(np.median(arrays["dice"])),
        "p05_dice": float(np.percentile(arrays["dice"], 5)),
        "mean_precision": float(np.mean(arrays["precision"])),
        "mean_recall": float(np.mean(arrays["recall"])),
        "mean_absolute_height_error_px": float(
            np.mean(arrays["absolute_height_error_px"])
        ),
        "max_absolute_height_error_px": int(
            np.max(arrays["absolute_height_error_px"])
        ),
    }


def _score_examples(
    examples: Sequence[MaskExample],
    model: Mapping[str, object],
    config: SpatialMaskConfig,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, np.ndarray]]:
    rows = []
    predictions = {}
    for example in examples:
        predicted = predict_mask(example.rgb, model, config)
        predictions[example.image_id] = predicted
        rows.append({"image_id": example.image_id, **score_mask(predicted, example.target_mask)})
    return rows, _aggregate(rows), predictions


def _gate(method: Mapping[str, object], control: Mapping[str, object]) -> dict[str, object]:
    checks = {
        "empty_prediction_count_is_zero": method["empty_prediction_count"] == 0,
        "mean_dice_at_least_0_85": method["mean_dice"] >= 0.85,
        "p05_dice_at_least_0_70": method["p05_dice"] >= 0.70,
        "mean_height_error_at_most_5_px": method["mean_absolute_height_error_px"] <= 5.0,
        "beats_train_mask_majority_control_mean_dice": method["mean_dice"] > control["mean_dice"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_png_new(path: Path, mask: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    data = buffer.getvalue()
    _write_new(path, data)
    return sha256_bytes(data)


def _repo_commit() -> str:
    root = Path(__file__).resolve().parents[4]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _environment() -> dict[str, str]:
    import PIL
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


_EXPECTED_DENOMINATORS = {
    "train_vials": 12,
    "train_stage_images": 30,
    "validation_vials": 3,
    "validation_stage_images": 6,
    "evaluation_vials": 3,
    "evaluation_stage_images": 6,
}


def run_validation_experiment(
    *,
    index_path: str | Path,
    merged_root: str | Path,
    preregistration_path: str | Path,
    output_dir: str | Path,
    expected_denominators: Mapping[str, int] = _EXPECTED_DENOMINATORS,
) -> dict[str, object]:
    """Fit the finite grid on train and select on validation only."""

    output = Path(output_dir)
    if os.path.lexists(output):
        raise FileExistsError(f"validation output already exists: {output}")
    prereg = Path(preregistration_path).resolve(strict=True)
    data = load_experiment_data(
        index_path=index_path,
        merged_root=merged_root,
        roles=("train", "validation"),
        expected_denominators=expected_denominators,
    )
    train = data.examples["train"]
    validation = data.examples["validation"]
    sampled = _sample_indices(train)
    models: dict[tuple[str, int], dict[str, object]] = {}
    model_bytes: dict[tuple[str, int], bytes] = {}
    for feature_set in _FEATURE_SETS:
        for leaf in _MIN_LEAVES:
            key = (feature_set, leaf)
            models[key] = fit_forest_model(
                train,
                sampled_indices=sampled,
                feature_set=feature_set,
                min_samples_leaf=leaf,
            )
            model_bytes[key] = canonical_json_bytes(models[key])
    majority = np.mean(
        np.stack([example.target_mask for example in train], axis=0), axis=0
    ) >= 0.5
    control_rows = [
        {"image_id": example.image_id, **score_mask(majority, example.target_mask)}
        for example in validation
    ]
    control_metrics = _aggregate(control_rows)
    candidates = []
    for config in candidate_configs():
        key = (config.feature_set, config.min_samples_leaf)
        rows, metrics, _ = _score_examples(validation, models[key], config)
        gate = _gate(metrics, control_metrics)
        candidates.append(
            {
                "config": asdict(config),
                "config_canonical_json": config.canonical_json(),
                "model_sha256": sha256_bytes(model_bytes[key]),
                "model_byte_length": len(model_bytes[key]),
                "validation_metrics": metrics,
                "validation_per_image": rows,
                "validation_gate": gate,
            }
        )
    candidates.sort(
        key=lambda row: (
            -int(bool(row["validation_gate"]["passed"])),
            float(row["validation_metrics"]["mean_absolute_height_error_px"]),
            -float(row["validation_metrics"]["p05_dice"]),
            -float(row["validation_metrics"]["mean_dice"]),
            int(row["model_byte_length"]),
            str(row["config_canonical_json"]),
        )
    )
    selected_record = candidates[0]
    selected = SpatialMaskConfig(**selected_record["config"])
    selected_key = (selected.feature_set, selected.min_samples_leaf)
    selected_model = models[selected_key]
    model_data = model_bytes[selected_key]
    config_artifact = {
        "schema_version": 1,
        "artifact_type": "spatial_rgb_mask_predictor_config",
        "status": "frozen_after_validation",
        "config": asdict(selected),
        "input_fields": ["rgb"],
        "threshold_source": "twin_validation_frozen",
    }
    config_data = canonical_json_bytes(config_artifact)
    train_rows, train_metrics, train_predictions = _score_examples(
        train, selected_model, selected
    )
    validation_rows, validation_metrics, validation_predictions = _score_examples(
        validation, selected_model, selected
    )
    gate = _gate(validation_metrics, control_metrics)
    result = {
        "schema_version": 1,
        "artifact_type": "spatial_rgb_mask_validation_result",
        "status": "validation_pass" if gate["passed"] else "validation_rejected",
        "seed": _SEED,
        "candidate_count": len(candidates),
        "source": {
            "index_path": os.fspath(Path(index_path).resolve(strict=True)),
            "index_sha256": data.index_sha256,
            "merged_root": os.fspath(Path(merged_root).resolve(strict=True)),
            "preregistration_path": os.fspath(prereg),
            "preregistration_sha256": sha256_bytes(prereg.read_bytes()),
        },
        "code_commit": _repo_commit(),
        "environment": _environment(),
        "invocation": {"argv": list(sys.argv), "cwd": os.fspath(Path.cwd())},
        "denominators": data.denominators,
        "selected_config": asdict(selected),
        "model_sha256": sha256_bytes(model_data),
        "config_sha256": sha256_bytes(config_data),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "validation_per_image": validation_rows,
        "train_mask_majority_control": {
            "definition": "pixelwise_train_mask_frequency_at_least_0_5",
            "validation_metrics": control_metrics,
            "validation_per_image": control_rows,
        },
        "validation_gate": gate,
        "evaluation_pixels_opened": False,
        "evaluation_status": "not_run_validation_gate_required",
        "claim_boundary": "Synthetic fixed-camera twin evidence only; no real-camera or plant claim.",
    }
    output.mkdir(parents=False, exist_ok=False)
    _write_new(output / "model.json", model_data)
    _write_new(output / "config.json", config_data)
    majority_sha = _write_png_new(output / "train-mask-majority-control.png", majority)
    prediction_locators = {}
    for role, examples, predictions in (
        ("train", train, train_predictions),
        ("validation", validation, validation_predictions),
    ):
        for example in examples:
            mask_path = (output / "predictions" / role / f"{example.image_id}.png").resolve()
            mask_sha = _write_png_new(mask_path, predictions[example.image_id])
            prediction_locators[example.image_id] = {
                "split_role": role,
                "rgb_path": os.fspath(example.rgb_path),
                "rgb_sha256": example.rgb_sha256,
                "mask_path": os.fspath(mask_path),
                "mask_sha256": mask_sha,
            }
    result["train_mask_majority_control"]["mask_sha256"] = majority_sha
    _write_new(output / "prediction-locators-train-validation.json", canonical_json_bytes(prediction_locators))
    _write_new(output / "candidate-results.json", canonical_json_bytes(candidates))
    _write_new(output / "train-per-image.json", canonical_json_bytes(train_rows))
    _write_new(output / "result.json", canonical_json_bytes(result))
    return result


def write_freeze_receipt(
    *, validation_dir: str | Path, output_path: str | Path
) -> dict[str, object]:
    """Bind committed code and selected artifacts before evaluation is opened."""

    directory = Path(validation_dir).resolve(strict=True)
    result_path = directory / "result.json"
    result = json.loads(result_path.read_bytes())
    if result.get("status") != "validation_pass" or not result.get("validation_gate", {}).get("passed"):
        raise ValueError("validation gate did not pass")
    commit = _repo_commit()
    if result.get("code_commit") != commit:
        raise ValueError("validation code commit differs from current commit")
    receipt = {
        "schema_version": 1,
        "artifact_type": "spatial_rgb_mask_evaluation_freeze",
        "status": "frozen_before_evaluation",
        "code_commit": commit,
        "validation_result_sha256": sha256_bytes(result_path.read_bytes()),
        "model_path": os.fspath((directory / "model.json").resolve(strict=True)),
        "model_sha256": sha256_bytes((directory / "model.json").read_bytes()),
        "config_path": os.fspath((directory / "config.json").resolve(strict=True)),
        "config_sha256": sha256_bytes((directory / "config.json").read_bytes()),
        "control_mask_path": os.fspath((directory / "train-mask-majority-control.png").resolve(strict=True)),
        "control_mask_sha256": sha256_bytes((directory / "train-mask-majority-control.png").read_bytes()),
        "candidate_results_sha256": sha256_bytes((directory / "candidate-results.json").read_bytes()),
        "preregistration_sha256": result["source"]["preregistration_sha256"],
        "index_sha256": result["source"]["index_sha256"],
        "input_fields": ["rgb"],
        "threshold_source": "twin_validation_frozen",
    }
    if receipt["model_sha256"] != result["model_sha256"] or receipt["config_sha256"] != result["config_sha256"]:
        raise ValueError("validation result artifact hash binding failed")
    _write_new(Path(output_path), canonical_json_bytes(receipt))
    return receipt


def _load_config(path: Path, expected_sha: str) -> SpatialMaskConfig:
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha:
        raise ValueError("config hash mismatch")
    artifact = json.loads(data)
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {"schema_version", "artifact_type", "status", "config", "input_fields", "threshold_source"}
        or artifact["schema_version"] != 1
        or artifact["artifact_type"] != "spatial_rgb_mask_predictor_config"
        or artifact["status"] != "frozen_after_validation"
        or artifact["input_fields"] != ["rgb"]
        or artifact["threshold_source"] != "twin_validation_frozen"
        or data != canonical_json_bytes(artifact)
    ):
        raise ValueError("config fields are invalid")
    return _validate_config(SpatialMaskConfig(**artifact["config"]))


def run_evaluation_once(
    *,
    index_path: str | Path,
    merged_root: str | Path,
    validation_dir: str | Path,
    freeze_receipt_path: str | Path,
    sentinel_path: str | Path,
    output_dir: str | Path,
    expected_denominators: Mapping[str, int] = _EXPECTED_DENOMINATORS,
) -> dict[str, object]:
    """Claim and execute the sole evaluation pass with frozen bytes."""

    output = Path(output_dir)
    if os.path.lexists(output):
        raise FileExistsError(f"evaluation output already exists: {output}")
    receipt_path = Path(freeze_receipt_path).resolve(strict=True)
    receipt_data = receipt_path.read_bytes()
    receipt = json.loads(receipt_data)
    required = {
        "schema_version", "artifact_type", "status", "code_commit",
        "validation_result_sha256", "model_path", "model_sha256", "config_path",
        "config_sha256", "control_mask_path", "control_mask_sha256",
        "candidate_results_sha256", "preregistration_sha256", "index_sha256",
        "input_fields", "threshold_source",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != required
        or receipt["status"] != "frozen_before_evaluation"
        or receipt["input_fields"] != ["rgb"]
        or receipt["threshold_source"] != "twin_validation_frozen"
        or receipt["code_commit"] != _repo_commit()
    ):
        raise ValueError("evaluation freeze receipt is invalid")
    validation = Path(validation_dir).resolve(strict=True)
    if sha256_bytes((validation / "result.json").read_bytes()) != receipt["validation_result_sha256"]:
        raise ValueError("validation result hash mismatch")
    if sha256_bytes((validation / "candidate-results.json").read_bytes()) != receipt["candidate_results_sha256"]:
        raise ValueError("candidate result hash mismatch")
    model = load_model(receipt["model_path"], receipt["model_sha256"])
    config = _load_config(Path(receipt["config_path"]), receipt["config_sha256"])
    control_data = Path(receipt["control_mask_path"]).read_bytes()
    if sha256_bytes(control_data) != receipt["control_mask_sha256"]:
        raise ValueError("control mask hash mismatch")
    with Image.open(BytesIO(control_data)) as image:
        if image.format != "PNG" or image.mode != "L":
            raise ValueError("control mask encoding is invalid")
        control = np.asarray(image, dtype=np.uint8).copy() == 255
    claim_evaluation_pass(sentinel_path)
    data = load_experiment_data(
        index_path=index_path,
        merged_root=merged_root,
        roles=("evaluation",),
        expected_denominators=expected_denominators,
    )
    if data.index_sha256 != receipt["index_sha256"]:
        raise ValueError("evaluation index hash mismatch")
    examples = data.examples["evaluation"]
    rows, metrics, predictions = _score_examples(examples, model, config)
    control_rows = [
        {"image_id": example.image_id, **score_mask(control, example.target_mask)}
        for example in examples
    ]
    control_metrics = _aggregate(control_rows)
    gate = _gate(metrics, control_metrics)
    result = {
        "schema_version": 1,
        "artifact_type": "spatial_rgb_mask_evaluation_result",
        "status": "evaluation_pass" if gate["passed"] else "evaluation_rejected",
        "one_pass_sentinel": os.fspath(Path(sentinel_path).resolve(strict=True)),
        "freeze_receipt_path": os.fspath(receipt_path),
        "freeze_receipt_sha256": sha256_bytes(receipt_data),
        "code_commit": receipt["code_commit"],
        "model_sha256": receipt["model_sha256"],
        "config_sha256": receipt["config_sha256"],
        "denominators": {"evaluation_vials": data.denominators["evaluation_vials"], "evaluation_stage_images": data.denominators["evaluation_stage_images"]},
        "evaluation_metrics": metrics,
        "evaluation_per_image": rows,
        "train_mask_majority_control": {
            "definition": "pixelwise_train_mask_frequency_at_least_0_5",
            "evaluation_metrics": control_metrics,
            "evaluation_per_image": control_rows,
        },
        "evaluation_gate": gate,
        "claim_boundary": "Synthetic fixed-camera twin evaluation only; no real-camera or plant claim.",
    }
    output.mkdir(parents=False, exist_ok=False)
    locators = {}
    for example in examples:
        mask_path = (output / "predictions" / "evaluation" / f"{example.image_id}.png").resolve()
        mask_sha = _write_png_new(mask_path, predictions[example.image_id])
        locators[example.image_id] = {
            "split_role": "evaluation",
            "rgb_path": os.fspath(example.rgb_path),
            "rgb_sha256": example.rgb_sha256,
            "mask_path": os.fspath(mask_path),
            "mask_sha256": mask_sha,
        }
    _write_new(output / "prediction-locators-evaluation.json", canonical_json_bytes(locators))
    _write_new(output / "result.json", canonical_json_bytes(result))
    return result


def publish_locator_artifact(
    *,
    validation_dir: str | Path,
    evaluation_dir: str | Path,
    freeze_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Combine already-produced masks without reopening any source RGB pixels."""

    validation = Path(validation_dir).resolve(strict=True)
    evaluation = Path(evaluation_dir).resolve(strict=True)
    evaluation_result = json.loads((evaluation / "result.json").read_bytes())
    if evaluation_result.get("status") != "evaluation_pass" or not evaluation_result.get("evaluation_gate", {}).get("passed"):
        raise ValueError("evaluation gate did not pass")
    receipt = json.loads(Path(freeze_receipt_path).resolve(strict=True).read_bytes())
    sources = [
        json.loads((validation / "prediction-locators-train-validation.json").read_bytes()),
        json.loads((evaluation / "prediction-locators-evaluation.json").read_bytes()),
    ]
    predictions: dict[str, dict[str, str]] = {}
    for source in sources:
        for image_id, locator in source.items():
            if image_id in predictions:
                raise ValueError("prediction image ID is duplicated")
            path = Path(locator["mask_path"])
            data = path.read_bytes()
            if sha256_bytes(data) != locator["mask_sha256"]:
                raise ValueError("prediction mask hash mismatch")
            with Image.open(BytesIO(data)) as image:
                if image.format != "PNG" or image.mode != "L":
                    raise ValueError("prediction mask encoding is invalid")
                mask = np.asarray(image, dtype=np.uint8)
                if not np.any(mask == 255) or not set(np.unique(mask).tolist()).issubset({0, 255}):
                    raise ValueError("prediction mask is empty or nonbinary")
            predictions[image_id] = {
                "rgb_path": locator["rgb_path"],
                "rgb_sha256": locator["rgb_sha256"],
                "mask_path": locator["mask_path"],
                "mask_sha256": locator["mask_sha256"],
            }
    if len(predictions) != 42:
        raise ValueError(f"prediction locator image count drifted: {len(predictions)}")
    rgb_paths = [locator["rgb_path"] for locator in predictions.values()]
    mask_paths = [locator["mask_path"] for locator in predictions.values()]
    if len(set(rgb_paths)) != 42 or len(set(mask_paths)) != 42:
        raise ValueError("prediction locator paths are not unique")
    artifact = build_locator_artifact(
        predictions=predictions,
        model_sha256=receipt["model_sha256"],
        config_sha256=receipt["config_sha256"],
        code_commit=receipt["code_commit"],
    )
    from ..masks import (
        build_trusted_predicted_mask_registry_v1,
        validate_rgb_mask_prediction_locators,
    )

    validate_rgb_mask_prediction_locators(artifact)
    registry = build_trusted_predicted_mask_registry_v1(artifact)
    if len(registry.image_ids) != 42:
        raise ValueError("trusted registry image count drifted")
    _write_new(Path(output_path), canonical_json_bytes(artifact))
    return artifact

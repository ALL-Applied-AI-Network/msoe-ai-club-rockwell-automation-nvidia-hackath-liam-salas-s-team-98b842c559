"""Production materialization boundary for accepted RGB-only liquid masks.

Inference consumes a prevalidated safe manifest containing camera RGB and a same-run
pre-pour RGB reference. It has no semantic-mask, recipe, label, palette,
composition, simulation-metadata, or video input.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image

from ..core.io import (
    fdopen_descriptor,
    open_descriptor,
    open_image,
    open_path,
    run_subprocess,
    sha256_file as _sha256,
)

from fluidmix.perception.baselines.empty_reference_rgb_predictor import (
    CHANGE_THRESHOLD,
    CLAMP_ROWS,
    IMAGE_SHAPE,
    MINIMUM_COMPONENT_AREA,
    MINIMUM_DENSE_ROWS,
    UPPER_MINIMUM_CHANGED_PIXELS,
    UPPER_MINIMUM_DENSE_ROWS,
    predict_visible_liquid,
)


PREDICTOR_CODE_COMMIT = "27c6699bae735c402a978687e34287520f61fedb"
PREDICTOR_SOURCE_SHA256 = "1e933c6eb8e6a729c820211c190aaa72f8d5ea203706e503484e3531800d80a1"
CNN_PREDICTOR_SOURCE_SHA256 = "ec9f4b589f088fe979f084ebff1ad51cb0fe0dd29cf12a64a7c993bab534be19"
CNN_CHECKPOINT_SHA256 = "3c32485f2306c9f302666f54a4923920863fa3a402cb82cbb25a1328d3a53fba"
CNN_CANDIDATE_RECEIPT_SHA256 = "27a4e49553f9fc0d34832e8f8ebc73f11fbf7dfa7e9c39f50353b1fae4b28727"
CNN_REPLAY_SHA256 = "a4b9d412ce063fa6449fbfcacf49b1b3ba3e7d64724fd93d96381c467ee4836a"
CNN_REVIEW_SHA256 = "17e1bb4809898c95b3af59cb6d187955a7bf36b4f6fffb39e342e07a59f2c8df"
CNN_SOURCE_COMMIT = "a5f29ea90272a3424d8c7de09dbf435509d28105"
THRESHOLD_SOURCE = "twin_validation_frozen"
ACCEPTED_RECOVERY_SHA256 = "5bb81e63ab53e2e704f1bfff7130038214fc8b5c05a28c86fc316dd62ea67374"
ACCEPTED_REVIEW_SHA256 = "77500b2e29d57e243103f02cfa5ebccb709338fde60622a66cdca3e699fb1c5f"
CORRECTED_V3_LOCATORS_SHA256 = "38963d90465391a18493375371efb42bb0442b120036ba296c9ea517aca73c4b"
CORRECTED_V3_VALIDATION_SHA256 = "3b34a81822a28cf6140f7b379583ace28b45d816514ef3e12b840cae022c54a2"
CORRECTED_V3_REQUEST_BUNDLE_SHA256 = "faba1fb330fa5c0aa37df4337a870924b8968492f6612bc954f830a1c7382a0a"
ACCEPTED_V3_SAFE_MANIFEST_SHA256 = "3c4871a710c02c8b71795f5486fa099aa0db22137c881b9b0882dfec480da723"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_IMAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REVISION_PATHS = (
    "src/fluidmix/perception/materializer.py",
    "cli/fluidmix/perception/materialize_empty_reference_predictions.py",
)

PREDICTOR_CONFIG = {
    "approach": "same_run_prepour_empty_reference_difference",
    "change_threshold": CHANGE_THRESHOLD,
    "clamp_rows": list(CLAMP_ROWS),
    "image_shape_hw": list(IMAGE_SHAPE),
    "minimum_component_area": MINIMUM_COMPONENT_AREA,
    "minimum_dense_rows": MINIMUM_DENSE_ROWS,
    "upper_minimum_changed_pixels": UPPER_MINIMUM_CHANGED_PIXELS,
    "upper_minimum_dense_rows": UPPER_MINIMUM_DENSE_ROWS,
}


class PredictedMaskMaterializationError(ValueError):
    """A production prediction input failed closed."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("utf-8") + b"\n"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


PREDICTOR_CONFIG_SHA256 = _canonical_sha256(PREDICTOR_CONFIG)


def _exact_object(value: object, fields: set[str], role: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PredictedMaskMaterializationError(f"{role} fields are invalid")
    return value


def _absolute_path(value: object, role: str) -> Path:
    try:
        path_value = os.fspath(value)
    except TypeError as error:
        raise PredictedMaskMaterializationError(f"{role} path is invalid") from error
    if not path_value:
        raise PredictedMaskMaterializationError(f"{role} path is invalid")
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink():
        raise PredictedMaskMaterializationError(
            f"{role} path must be absolute and symlink-free"
        )
    return path


def _digest(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PredictedMaskMaterializationError(f"{role} hash is invalid")
    return value


def _verified_path(value: object, digest: object, role: str) -> Path:
    path = _absolute_path(value, role)
    expected = _digest(digest, role)
    if not path.is_file() or _sha256(path) != expected:
        raise PredictedMaskMaterializationError(f"{role} SHA-256 mismatch")
    return path


def _load_rgb(path: Path, digest: str, role: str) -> np.ndarray:
    if _sha256(path) != digest:
        raise PredictedMaskMaterializationError(f"{role} SHA-256 mismatch")
    with open_image(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (640, 480):
            raise PredictedMaskMaterializationError(
                f"{role} must be an exact 640x480 RGB PNG"
            )
        image.load()
        return np.asarray(image, dtype=np.uint8).copy()


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = open_descriptor(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with fdopen_descriptor(descriptor, "wb") as stream:
            stream.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _git_output(repository_root: Path, *arguments: str) -> bytes:
    result = run_subprocess(
        ["git", "-C", os.fspath(repository_root), *arguments],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise PredictedMaskMaterializationError(
            f"materializer Git verification failed: {detail or arguments[0]}"
        )
    return result.stdout


def _verify_runtime_revision(materializer_commit: str) -> dict[str, object]:
    """Bind the executing tracked bytes to the clean repository HEAD."""

    repository_root = _REPOSITORY_ROOT
    observed_root = Path(
        _git_output(repository_root, "rev-parse", "--show-toplevel")
        .decode("utf-8")
        .strip()
    ).resolve()
    if observed_root != repository_root:
        raise PredictedMaskMaterializationError(
            "materializer repository root differs from Git worktree"
        )
    head = _git_output(repository_root, "rev-parse", "HEAD").decode("ascii").strip()
    if materializer_commit != head:
        raise PredictedMaskMaterializationError(
            "materializer commit claim does not match Git HEAD"
        )
    dirty = _git_output(
        repository_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
        "--",
        *_REVISION_PATHS,
    )
    if dirty:
        raise PredictedMaskMaterializationError(
            "materializer Git worktree has dirty relevant files"
        )
    files: dict[str, dict[str, str]] = {}
    for relative in _REVISION_PATHS:
        path = repository_root / relative
        if path.is_symlink() or not path.is_file():
            raise PredictedMaskMaterializationError(
                f"materializer tracked file is missing or symlinked: {relative}"
            )
        worktree_bytes = path.read_bytes()
        head_bytes = _git_output(repository_root, "show", f"HEAD:{relative}")
        if worktree_bytes != head_bytes:
            raise PredictedMaskMaterializationError(
                f"materializer tracked bytes differ from Git HEAD: {relative}"
            )
        files[relative] = {
            "path": os.fspath(path),
            "sha256": hashlib.sha256(worktree_bytes).hexdigest(),
        }
    return {
        "code_commit": head,
        "repository_root": os.fspath(repository_root),
        "tracked_files": files,
        "relevant_worktree_clean": True,
    }


def _save_mask_new(path: Path, mask: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path, format="PNG")


def _load_frozen_cnn(
    *, checkpoint_path: str | Path, replay_path: str | Path, review_path: str | Path
) -> tuple[object, dict[str, object], dict[str, object], dict[str, str]]:
    """Load only the reviewed frozen CNN/checkpoint/threshold identity."""

    checkpoint = _verified_path(checkpoint_path, CNN_CHECKPOINT_SHA256, "CNN checkpoint")
    replay_file = _verified_path(replay_path, CNN_REPLAY_SHA256, "CNN deterministic replay")
    review_file = _verified_path(review_path, CNN_REVIEW_SHA256, "CNN independent review")
    try:
        replay = json.loads(replay_file.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictedMaskMaterializationError("CNN replay is not strict JSON") from error
    if not isinstance(replay, Mapping):
        raise PredictedMaskMaterializationError("CNN replay is invalid")
    sources = replay.get("source_bindings")
    scoring = replay.get("scoring")
    if (
        replay.get("artifact_type") != "wp5_cnn_deterministic_bounds_replay"
        or replay.get("status") != "swap_review_ready"
        or replay.get("checkpoint_frozen") is not True
        or replay.get("threshold_frozen") != 0.5
        or not isinstance(scoring, Mapping)
        or not isinstance(scoring.get("all"), Mapping)
        or not isinstance(scoring.get("cleaned"), Mapping)
        or scoring["all"].get("all_gates_pass") is not True
        or scoring["cleaned"].get("all_gates_pass") is not True
        or scoring["all"].get("decision") != "replace_approach_a"
        or scoring["cleaned"].get("decision") != "replace_approach_a"
        or not isinstance(sources, Mapping)
    ):
        raise PredictedMaskMaterializationError("CNN replay did not authorize the frozen swap")
    candidate_binding = sources.get("candidate_receipt")
    checkpoint_binding = sources.get("checkpoint")
    try:
        bound_checkpoint = (
            Path(str(checkpoint_binding.get("path"))).resolve(strict=True)
            if isinstance(checkpoint_binding, Mapping)
            else None
        )
    except OSError as error:
        raise PredictedMaskMaterializationError("CNN replay checkpoint is unavailable") from error
    if (
        not isinstance(candidate_binding, Mapping)
        or candidate_binding.get("sha256") != CNN_CANDIDATE_RECEIPT_SHA256
        or not isinstance(checkpoint_binding, Mapping)
        or checkpoint_binding.get("sha256") != CNN_CHECKPOINT_SHA256
        or bound_checkpoint != checkpoint
    ):
        raise PredictedMaskMaterializationError("CNN replay lineage is invalid")
    candidate_file = _verified_path(
        candidate_binding.get("path"), CNN_CANDIDATE_RECEIPT_SHA256, "CNN candidate receipt"
    )
    try:
        candidate = json.loads(candidate_file.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictedMaskMaterializationError("CNN candidate receipt is not strict JSON") from error
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("artifact_type") != "wp5_cnn_frozen_candidate"
        or candidate.get("source_commit") != CNN_SOURCE_COMMIT
        or candidate.get("checkpoint_sha256") != CNN_CHECKPOINT_SHA256
        or candidate.get("threshold") != 0.5
    ):
        raise PredictedMaskMaterializationError("CNN candidate identity is invalid")
    if not review_file.read_text(encoding="utf-8").lstrip().startswith("# Independent"):
        raise PredictedMaskMaterializationError("CNN independent review format is invalid")
    try:
        import torch
        from .cnn.cnn_predictor import ResNet18UNet

        model = ResNet18UNet(weights=None)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
    except (OSError, RuntimeError, ValueError) as error:
        raise PredictedMaskMaterializationError("CNN checkpoint could not be loaded") from error
    config = {
        "architecture": "six-channel-resnet18-unet",
        "candidate_receipt_sha256": CNN_CANDIDATE_RECEIPT_SHA256,
        "checkpoint_sha256": CNN_CHECKPOINT_SHA256,
        "deterministic_replay_sha256": CNN_REPLAY_SHA256,
        "independent_review_sha256": CNN_REVIEW_SHA256,
        "probability_threshold": 0.5,
        "postprocessing": "unchanged_approach_a_vial_roi_and_clamp_raster",
    }
    producer = {
        "producer_id": "frozen-wp5-six-channel-cnn",
        "producer_kind": "rgb_only_liquid_segmenter",
        "model_id": "resnet18-unet-current-empty-rgb",
        "model_sha256": CNN_CHECKPOINT_SHA256,
        "config_id": "wp5-cnn-six-channel-frozen-v1",
        "config_sha256": _canonical_sha256(config),
        "code_commit": CNN_SOURCE_COMMIT,
        "threshold_source": THRESHOLD_SOURCE,
        "training_data_origin": "twin",
        "input_fields": ["rgb"],
    }
    accepted = {
        "recovered_result_path": os.fspath(replay_file),
        "recovered_result_sha256": CNN_REPLAY_SHA256,
        "independent_review_path": os.fspath(review_file),
        "independent_review_sha256": CNN_REVIEW_SHA256,
    }
    return model, producer, config, accepted


def _predict_cnn_pair(model: object, rgb: np.ndarray, reference: np.ndarray) -> object:
    from .cnn.cnn_predictor import predict_pair

    return predict_pair(model, rgb, reference, threshold=0.5, device="cpu")


def _validate_safe_manifest(
    value: object,
) -> tuple[list[Mapping[str, object]], Mapping[str, object], Mapping[str, object]]:
    artifact = _exact_object(
        value,
        {
            "schema_version",
            "artifact_type",
            "data_origin",
            "input_fields",
            "threshold_source",
            "split_version",
            "accepted_evidence",
            "source_bindings",
            "records",
        },
        "safe manifest",
    )
    if (
        artifact["schema_version"] != 1
        or artifact["artifact_type"] != "empty_reference_rgb_production_inputs"
        or artifact["data_origin"] != "twin"
        or artifact["input_fields"] != ["rgb"]
        or artifact["threshold_source"] != THRESHOLD_SOURCE
        or artifact["split_version"] != "whole-run-varied-v3"
    ):
        raise PredictedMaskMaterializationError("safe manifest identity drifted")
    accepted = _exact_object(
        artifact["accepted_evidence"],
        {
            "recovered_result_path",
            "recovered_result_sha256",
            "independent_review_path",
            "independent_review_sha256",
        },
        "accepted evidence",
    )
    sources = _exact_object(
        artifact["source_bindings"],
        {
            "image_locators_path",
            "image_locators_sha256",
            "downstream_validation_receipt_path",
            "downstream_validation_receipt_sha256",
            "request_bundle_path",
            "request_bundle_sha256",
        },
        "source bindings",
    )
    records = artifact["records"]
    if not isinstance(records, list) or not records:
        raise PredictedMaskMaterializationError("safe manifest records are invalid")
    return records, accepted, sources


_RECORD_FIELDS = {
    "image_id",
    "batch_id",
    "run_id",
    "vial_id",
    "split_role",
    "stage_id",
    "stage_ordinal",
    "rgb_path",
    "rgb_sha256",
    "reference_rgb_path",
    "reference_rgb_sha256",
    "source_capture_manifest_path",
    "source_capture_manifest_sha256",
    "source_receipt_path",
    "source_receipt_sha256",
}


def materialize_predictions(
    *,
    safe_manifest_path: str | Path,
    output_dir: str | Path,
    materializer_commit: str,
    require_accepted_v3: bool = True,
    cnn_checkpoint_path: str | Path | None = None,
    cnn_replay_path: str | Path | None = None,
    cnn_review_path: str | Path | None = None,
) -> dict[str, object]:
    """Publish schema-v2 locators and PNG masks from RGB-only safe inputs."""

    cnn_bindings = (cnn_checkpoint_path, cnn_replay_path, cnn_review_path)
    if any(value is not None for value in cnn_bindings) and not all(
        value is not None for value in cnn_bindings
    ):
        raise PredictedMaskMaterializationError("CNN binding set is incomplete")
    use_cnn = all(value is not None for value in cnn_bindings)
    if not isinstance(materializer_commit, str) or _COMMIT.fullmatch(materializer_commit) is None:
        raise PredictedMaskMaterializationError(
            "materializer commit must be 40 lowercase hex"
        )
    materializer_provenance = _verify_runtime_revision(materializer_commit)
    predictor_source_path = (
        _REPOSITORY_ROOT / "src" / "fluidmix" / "perception" / "cnn" / "cnn_predictor.py"
        if use_cnn
        else _REPOSITORY_ROOT / "src" / "fluidmix" / "perception" / "baselines"
        / "empty_reference_rgb_predictor.py"
    )
    expected_predictor_source = (
        CNN_PREDICTOR_SOURCE_SHA256 if use_cnn else PREDICTOR_SOURCE_SHA256
    )
    if _sha256(predictor_source_path) != expected_predictor_source:
        raise PredictedMaskMaterializationError(
            "predictor source differs from the accepted implementation"
        )
    safe_path = Path(safe_manifest_path)
    if safe_path.is_symlink() or not safe_path.is_file():
        raise PredictedMaskMaterializationError("safe manifest must be a regular file")
    safe_sha = _sha256(safe_path)
    try:
        manifest = json.loads(safe_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictedMaskMaterializationError("safe manifest is not strict JSON") from error
    records, accepted, sources = _validate_safe_manifest(manifest)
    cnn_model: object | None = None
    cnn_producer: dict[str, object] | None = None
    cnn_config: dict[str, object] | None = None
    if use_cnn:
        assert cnn_checkpoint_path is not None
        assert cnn_replay_path is not None
        assert cnn_review_path is not None
        cnn_model, cnn_producer, cnn_config, accepted = _load_frozen_cnn(
            checkpoint_path=cnn_checkpoint_path,
            replay_path=cnn_replay_path,
            review_path=cnn_review_path,
        )

    if not isinstance(require_accepted_v3, bool):
        raise TypeError("require_accepted_v3 must be bool")
    if require_accepted_v3:
        if safe_sha != ACCEPTED_V3_SAFE_MANIFEST_SHA256:
            raise PredictedMaskMaterializationError(
                "safe manifest differs from the exact accepted V3 projection"
            )
        expected_sources = {
            "image_locators_sha256": CORRECTED_V3_LOCATORS_SHA256,
            "downstream_validation_receipt_sha256": CORRECTED_V3_VALIDATION_SHA256,
            "request_bundle_sha256": CORRECTED_V3_REQUEST_BUNDLE_SHA256,
        }
        if any(sources.get(key) != value for key, value in expected_sources.items()):
            raise PredictedMaskMaterializationError(
                "safe manifest source hashes differ from accepted V3"
            )
        split_counts: dict[str, int] = {}
        run_ids: set[object] = set()
        vial_ids: set[tuple[object, object]] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise PredictedMaskMaterializationError("safe input record is invalid")
            role = record.get("split_role")
            if not isinstance(role, str):
                raise PredictedMaskMaterializationError("split_role is invalid")
            split_counts[role] = split_counts.get(role, 0) + 1
            run_ids.add(record.get("run_id"))
            vial_ids.add((record.get("batch_id"), record.get("vial_id")))
        if (
            len(records) != 42
            or split_counts != {"train": 30, "validation": 6, "evaluation": 6}
            or len(run_ids) != 6
            or len(vial_ids) != 18
        ):
            raise PredictedMaskMaterializationError(
                "accepted V3 image/split/run/vial denominator drifted"
            )

    for role, path_field, sha_field in (
        ("accepted recovered result", "recovered_result_path", "recovered_result_sha256"),
        ("accepted independent review", "independent_review_path", "independent_review_sha256"),
    ):
        _verified_path(accepted[path_field], accepted[sha_field], role)
    for role in ("image_locators", "downstream_validation_receipt", "request_bundle"):
        if require_accepted_v3:
            _verified_path(
                sources[f"{role}_path"], sources[f"{role}_sha256"], role
            )
        else:
            _absolute_path(sources[f"{role}_path"], role)
            _digest(sources[f"{role}_sha256"], role)

    if require_accepted_v3:
        verified_provenance: set[tuple[str, str]] = set()
        for raw in records:
            assert isinstance(raw, Mapping)
            for role, path_field, sha_field in (
                (
                    "source capture manifest",
                    "source_capture_manifest_path",
                    "source_capture_manifest_sha256",
                ),
                ("source renderer receipt", "source_receipt_path", "source_receipt_sha256"),
            ):
                key = (str(raw.get(path_field)), str(raw.get(sha_field)))
                if key not in verified_provenance:
                    _verified_path(raw.get(path_field), raw.get(sha_field), role)
                    verified_provenance.add(key)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)

    prepared: list[dict[str, object]] = []
    image_ids: set[str] = set()
    identities: set[tuple[str, str, str, int]] = set()
    reference_by_case: dict[str, tuple[str, str]] = {}
    for raw in records:
        record = _exact_object(raw, _RECORD_FIELDS, "safe input record")
        image_id = record["image_id"]
        if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
            raise PredictedMaskMaterializationError("image ID is unsafe")
        if image_id in image_ids:
            raise PredictedMaskMaterializationError("duplicate image ID")
        image_ids.add(image_id)
        for field in ("batch_id", "run_id", "vial_id", "split_role", "stage_id"):
            if not isinstance(record[field], str) or not record[field]:
                raise PredictedMaskMaterializationError(f"{field} is invalid")
        if record["split_role"] not in {"train", "validation", "evaluation"}:
            raise PredictedMaskMaterializationError("split_role is invalid")
        ordinal = record["stage_ordinal"]
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise PredictedMaskMaterializationError("stage_ordinal is invalid")
        identity = (
            str(record["batch_id"]),
            str(record["vial_id"]),
            str(record["stage_id"]),
            ordinal,
        )
        if identity in identities:
            raise PredictedMaskMaterializationError("duplicate stage identity")
        identities.add(identity)

        rgb_path = _absolute_path(record["rgb_path"], "stage RGB")
        rgb_sha = _digest(record["rgb_sha256"], "stage RGB")
        reference_path = _absolute_path(record["reference_rgb_path"], "pre-pour RGB")
        reference_sha = _digest(record["reference_rgb_sha256"], "pre-pour RGB")
        if rgb_path == reference_path:
            raise PredictedMaskMaterializationError("stage RGB aliases pre-pour RGB")
        case_key = str(record["source_receipt_path"])
        reference_binding = (os.fspath(reference_path), reference_sha)
        previous = reference_by_case.setdefault(case_key, reference_binding)
        if previous != reference_binding:
            raise PredictedMaskMaterializationError(
                "one case has multiple pre-pour references"
            )

        rgb = _load_rgb(rgb_path, rgb_sha, "stage RGB")
        reference = _load_rgb(reference_path, reference_sha, "pre-pour RGB")
        prediction = (
            _predict_cnn_pair(cnn_model, rgb, reference)
            if use_cnn
            else predict_visible_liquid(rgb, reference)
        )
        prepared.append(
            {
                "record": record,
                "image_id": image_id,
                "ordinal": ordinal,
                "rgb_path": rgb_path,
                "rgb_sha256": rgb_sha,
                "reference_path": reference_path,
                "reference_sha256": reference_sha,
                "prediction": prediction,
            }
        )

    output.mkdir(parents=True, exist_ok=False)
    masks_dir = output / "masks"
    masks_dir.mkdir()
    safe_copy_path = (output / "safe-inputs.json").resolve()
    _write_new(safe_copy_path, safe_path.read_bytes())

    images: dict[str, dict[str, str]] = {}
    receipt_records: list[dict[str, object]] = []
    for item in prepared:
        record = item["record"]
        assert isinstance(record, Mapping)
        image_id = str(item["image_id"])
        ordinal = int(item["ordinal"])
        rgb_path = item["rgb_path"]
        reference_path = item["reference_path"]
        prediction = item["prediction"]
        assert isinstance(rgb_path, Path)
        assert isinstance(reference_path, Path)
        mask_path = (masks_dir / f"{image_id}.png").resolve()
        _save_mask_new(mask_path, prediction.mask)
        mask_sha = _sha256(mask_path)
        images[image_id] = {
            "rgb_path": os.fspath(rgb_path),
            "rgb_sha256": str(item["rgb_sha256"]),
            "mask_path": os.fspath(mask_path),
            "mask_sha256": mask_sha,
        }
        receipt_records.append(
            {
                "image_id": image_id,
                "batch_id": record["batch_id"],
                "run_id": record["run_id"],
                "vial_id": record["vial_id"],
                "split_role": record["split_role"],
                "stage_id": record["stage_id"],
                "stage_ordinal": ordinal,
                "rgb_path": os.fspath(rgb_path),
                "rgb_sha256": item["rgb_sha256"],
                "reference_rgb_path": os.fspath(reference_path),
                "reference_rgb_sha256": item["reference_sha256"],
                "mask_path": os.fspath(mask_path),
                "mask_sha256": mask_sha,
                "prediction_status": prediction.status,
                "surface_row": prediction.surface_row,
                "visible_height_px": prediction.visible_height_px,
                "changed_pixel_count": prediction.changed_pixel_count,
                "source_capture_manifest_path": record["source_capture_manifest_path"],
                "source_capture_manifest_sha256": record[
                    "source_capture_manifest_sha256"
                ],
                "source_receipt_path": record["source_receipt_path"],
                "source_receipt_sha256": record["source_receipt_sha256"],
                "history_eligible": record["split_role"] != "evaluation",
                "history_eligibility_reason": (
                    "accepted_non_evaluation_partition"
                    if record["split_role"] != "evaluation"
                    else "evaluation_partition_excluded"
                ),
            }
        )

    producer = cnn_producer or {
        "producer_id": "accepted-empty-reference-rgb-predictor",
        "producer_kind": "rgb_only_liquid_segmenter",
        "model_id": "empty-reference-rgb-approach-a",
        "model_sha256": PREDICTOR_SOURCE_SHA256,
        "config_id": "empty-reference-rgb-approach-a-frozen-v1",
        "config_sha256": PREDICTOR_CONFIG_SHA256,
        "code_commit": PREDICTOR_CODE_COMMIT,
        "threshold_source": THRESHOLD_SOURCE,
        "training_data_origin": "twin",
        "input_fields": ["rgb"],
    }
    predictor_config = cnn_config or PREDICTOR_CONFIG
    history_eligible_ids = {
        str(row["image_id"])
        for row in receipt_records
        if row["history_eligible"]
    }
    locator = {
        "schema_version": 2,
        "artifact_type": "rgb_liquid_mask_prediction_locators",
        "data_origin": "twin",
        "mask_source": "rgb_only_prediction",
        "producer": producer,
        "images": {
            key: images[key]
            for key in sorted(images)
            if key in history_eligible_ids
        },
    }
    locator_path = (output / "prediction-locators.json").resolve()
    _write_new(locator_path, _canonical_bytes(locator))
    locator_sha = _sha256(locator_path)

    split_counts: dict[str, int] = {}
    for record in receipt_records:
        role = str(record["split_role"])
        split_counts[role] = split_counts.get(role, 0) + 1
    receipt = {
        "schema_version": 1,
        "artifact_type": "empty_reference_rgb_prediction_materialization_receipt",
        "status": "materialized",
        "materializer": materializer_provenance,
        "predictor": {**producer, "config": predictor_config},
        "accepted_evidence": dict(accepted),
        "safe_manifest": {
            "path": os.fspath(safe_copy_path),
            "sha256": safe_sha,
        },
        "source_bindings": dict(sources),
        "prediction_locators": {
            "path": os.fspath(locator_path),
            "sha256": locator_sha,
        },
        "denominators": {
            "images": len(images),
            "vials": len(
                {(row["batch_id"], row["vial_id"]) for row in receipt_records}
            ),
            "runs": len({row["run_id"] for row in receipt_records}),
            "split_images": dict(sorted(split_counts.items())),
            "history_eligible_images": sum(
                bool(row["history_eligible"]) for row in receipt_records
            ),
            "history_excluded_evaluation_images": sum(
                row["split_role"] == "evaluation" for row in receipt_records
            ),
        },
        "inference_inputs": {
            "input_fields": ["rgb"],
            "oracle_masks_opened": 0,
            "semantic_masks_opened": 0,
            "sim_metadata_inputs": 0,
            "label_inputs": 0,
            "palette_inputs": 0,
            "composition_inputs": 0,
            "recipe_ml_inputs": 0,
            "video_inputs": 0,
        },
        "recipe_invariance": {
            "recipe_fields_present_in_safe_manifest": False,
            "programmed_ml_argument_supplied": False,
            "accepted_recovery_check": (
                "cnn_model_inputs_exclude_recipe_ml"
                if use_cnn
                else "programmed_ml_stage_outputs_invariant"
            ),
        },
        "records": sorted(receipt_records, key=lambda row: str(row["image_id"])),
    }
    receipt_path = (output / "materialization-receipt.json").resolve()
    _write_new(receipt_path, _canonical_bytes(receipt))
    return receipt


def _load_json_with_hash(path: Path, expected_sha256: str, role: str) -> Mapping[str, object]:
    verified = _verified_path(os.fspath(path), expected_sha256, role)
    try:
        value = json.loads(verified.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PredictedMaskMaterializationError(f"{role} is not strict JSON") from error
    if not isinstance(value, Mapping):
        raise PredictedMaskMaterializationError(f"{role} must be an object")
    return value


def prepare_v3_safe_manifest(
    *,
    image_locators_path: str | Path,
    downstream_validation_receipt_path: str | Path,
    request_bundle_path: str | Path,
    recovered_result_path: str | Path,
    independent_review_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Project exact corrected V3 metadata into a 42-row RGB-only manifest.

    This metadata preparation reads renderer receipts only to bind the same-run pre-pour
    RGB. It never opens the semantic mask paths carried by the legacy capture locator.
    """

    locator_path = Path(image_locators_path)
    validation_path = Path(downstream_validation_receipt_path)
    bundle_path = Path(request_bundle_path)
    recovery_path = Path(recovered_result_path)
    review_path = Path(independent_review_path)
    locators = _load_json_with_hash(
        locator_path, CORRECTED_V3_LOCATORS_SHA256, "corrected V3 image locators"
    )
    validation = _load_json_with_hash(
        validation_path,
        CORRECTED_V3_VALIDATION_SHA256,
        "corrected V3 downstream validation receipt",
    )
    bundle = _load_json_with_hash(
        bundle_path,
        CORRECTED_V3_REQUEST_BUNDLE_SHA256,
        "corrected V3 request bundle",
    )
    _verified_path(
        os.fspath(recovery_path), ACCEPTED_RECOVERY_SHA256, "accepted recovered result"
    )
    _verified_path(
        os.fspath(review_path), ACCEPTED_REVIEW_SHA256, "accepted independent review"
    )
    if (
        locators.get("artifact_type") != "f2_varied_image_locators"
        or locators.get("schema_version") != 1
        or locators.get("split_version") != "whole-run-varied-v3"
        or locators.get("status") != "published"
        or validation.get("artifact_type")
        != "f2_varied_v3_downstream_validation_receipt"
        or validation.get("schema_version") != 2
        or validation.get("status") != "pass"
        or bundle.get("artifact_type") != "f2_varied_stage_publish_request_bundle"
        or bundle.get("schema_version") != 3
        or bundle.get("status") != "materialized"
    ):
        raise PredictedMaskMaterializationError("corrected V3 artifact identity drifted")
    validation_binding = locators.get("source_validation_receipt")
    if not isinstance(validation_binding, Mapping) or (
        validation_binding.get("path") != os.fspath(validation_path)
        or validation_binding.get("sha256") != CORRECTED_V3_VALIDATION_SHA256
    ):
        raise PredictedMaskMaterializationError("image locator validation binding drifted")

    source_manifests = locators.get("source_manifests")
    images = locators.get("images")
    requests = bundle.get("requests")
    if (
        not isinstance(source_manifests, list)
        or len(source_manifests) != 6
        or not isinstance(images, Mapping)
        or len(images) != 42
        or not isinstance(requests, list)
        or len(requests) != 6
    ):
        raise PredictedMaskMaterializationError("corrected V3 denominator drifted")

    validation_files = validation.get("files")
    if not isinstance(validation_files, list):
        raise PredictedMaskMaterializationError("validation file inventory is invalid")
    validated_hashes = {
        str(row.get("path")): str(row.get("sha256"))
        for row in validation_files
        if isinstance(row, Mapping)
    }
    capture_rows: dict[str, Mapping[str, object]] = {}
    capture_binding_by_batch: dict[str, tuple[str, str]] = {}
    for source in source_manifests:
        if not isinstance(source, Mapping):
            raise PredictedMaskMaterializationError("source manifest binding is invalid")
        path = _absolute_path(source.get("path"), "source capture manifest")
        digest = _digest(source.get("sha256"), "source capture manifest")
        capture = _load_json_with_hash(path, digest, "source capture manifest")
        batch_id = source.get("batch_id")
        if (
            not isinstance(batch_id, str)
            or capture.get("artifact_type") != "f2_stage_capture_manifest"
            or capture.get("schema_version") != 3
            or capture.get("status") != "accepted"
            or capture.get("batch_id") != batch_id
            or capture.get("run_id") != source.get("run_id")
            or capture.get("split_role") != source.get("split_role")
            or capture.get("split_version") != "whole-run-varied-v3"
        ):
            raise PredictedMaskMaterializationError("capture manifest identity drifted")
        relative = os.fspath(path.relative_to(validation_path.parent))
        if validated_hashes.get(relative) != digest:
            raise PredictedMaskMaterializationError("capture manifest is absent from validation receipt")
        capture_binding_by_batch[batch_id] = (os.fspath(path), digest)
        rows = capture.get("rows")
        if not isinstance(rows, list):
            raise PredictedMaskMaterializationError("capture rows are invalid")
        for row in rows:
            if not isinstance(row, Mapping):
                raise PredictedMaskMaterializationError("capture row is invalid")
            image_id = f"{batch_id}-{row.get('vial_id')}-{row.get('event_id')}"
            if image_id in capture_rows:
                raise PredictedMaskMaterializationError("duplicate capture image ID")
            capture_rows[image_id] = row

    case_by_batch_vial: dict[tuple[str, str], Mapping[str, object]] = {}
    for request_binding in requests:
        if not isinstance(request_binding, Mapping):
            raise PredictedMaskMaterializationError("request binding is invalid")
        relative = request_binding.get("path")
        digest = _digest(request_binding.get("sha256"), "publish request")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise PredictedMaskMaterializationError("publish request path is invalid")
        request_path = bundle_path.parent / relative
        request = _load_json_with_hash(request_path, digest, "publish request")
        validation_relative = os.fspath(request_path.relative_to(validation_path.parent))
        if validated_hashes.get(validation_relative) != digest:
            raise PredictedMaskMaterializationError("publish request is absent from validation receipt")
        batch_id = request.get("batch_id")
        vials = request.get("vials")
        if (
            request.get("artifact_type") != "f2_varied_stage_batch_publish_request"
            or request.get("schema_version") != 3
            or request.get("split_version") != "whole-run-varied-v3"
            or batch_id != request_binding.get("batch_id")
            or not isinstance(vials, list)
            or len(vials) != 3
        ):
            raise PredictedMaskMaterializationError("publish request identity drifted")
        for vial in vials:
            if not isinstance(vial, Mapping) or not isinstance(vial.get("vial_id"), str):
                raise PredictedMaskMaterializationError("publish request vial is invalid")
            key = (str(batch_id), str(vial["vial_id"]))
            if key in case_by_batch_vial:
                raise PredictedMaskMaterializationError("duplicate batch/vial request")
            case_by_batch_vial[key] = vial

    records: list[dict[str, object]] = []
    receipt_cache: dict[str, tuple[str, str]] = {}
    for image_id in sorted(images):
        locator = images[image_id]
        capture = capture_rows.get(str(image_id))
        if not isinstance(locator, Mapping) or capture is None:
            raise PredictedMaskMaterializationError("image locator/capture mapping is incomplete")
        rgb = capture.get("rgb")
        if not isinstance(rgb, Mapping):
            raise PredictedMaskMaterializationError("capture RGB binding is invalid")
        expected_identity = {
            "batch_id": locator.get("batch_id"),
            "vial_id": locator.get("vial_id"),
            "stage_id": locator.get("stage_id"),
            "stage_ordinal": locator.get("stage_ordinal"),
        }
        if (
            str(image_id)
            != f"{expected_identity['batch_id']}-{expected_identity['vial_id']}-{expected_identity['stage_id']}"
            or capture.get("vial_id") != expected_identity["vial_id"]
            or capture.get("event_id") != expected_identity["stage_id"]
            or capture.get("stage_ordinal") != expected_identity["stage_ordinal"]
            or rgb.get("path") != locator.get("rgb_path")
            or rgb.get("sha256") != locator.get("rgb_sha256")
            or locator.get("source_manifest_sha256")
            != capture_binding_by_batch[str(locator.get("batch_id"))][1]
        ):
            raise PredictedMaskMaterializationError("image locator differs from capture manifest")
        vial = case_by_batch_vial.get(
            (str(locator.get("batch_id")), str(locator.get("vial_id")))
        )
        if vial is None:
            raise PredictedMaskMaterializationError("image has no batch/vial receipt mapping")
        receipt_path = _absolute_path(vial.get("receipt_path"), "source renderer receipt")
        receipt_sha = _digest(vial.get("receipt_sha256"), "source renderer receipt")
        cache_key = os.fspath(receipt_path)
        cached = receipt_cache.get(cache_key)
        if cached is None:
            receipt = _load_json_with_hash(receipt_path, receipt_sha, "source renderer receipt")
            parity = receipt.get("pre_pour_parity")
            if (
                not isinstance(parity, Mapping)
                or parity.get("passed") is not True
                or parity.get("f2_prefill_path")
                != "pre-pour-parity/f2-prefill-time-0001.png"
            ):
                raise PredictedMaskMaterializationError("same-run pre-pour binding is invalid")
            reference_path = receipt_path.parent / str(parity["f2_prefill_path"])
            reference_sha = _digest(parity.get("f2_prefill_sha256"), "same-run pre-pour RGB")
            _verified_path(os.fspath(reference_path), reference_sha, "same-run pre-pour RGB")
            cached = (os.fspath(reference_path), reference_sha)
            receipt_cache[cache_key] = cached
        records.append(
            {
                "image_id": str(image_id),
                "batch_id": locator["batch_id"],
                "run_id": locator["run_id"],
                "vial_id": locator["vial_id"],
                "split_role": locator["split_role"],
                "stage_id": locator["stage_id"],
                "stage_ordinal": locator["stage_ordinal"],
                "rgb_path": locator["rgb_path"],
                "rgb_sha256": locator["rgb_sha256"],
                "reference_rgb_path": cached[0],
                "reference_rgb_sha256": cached[1],
                "source_capture_manifest_path": capture_binding_by_batch[
                    str(locator["batch_id"])
                ][0],
                "source_capture_manifest_sha256": capture_binding_by_batch[
                    str(locator["batch_id"])
                ][1],
                "source_receipt_path": os.fspath(receipt_path),
                "source_receipt_sha256": receipt_sha,
            }
        )

    if len(records) != 42 or len(case_by_batch_vial) != 18 or len(receipt_cache) != 18:
        raise PredictedMaskMaterializationError("safe V3 mapping denominator drifted")
    artifact = {
        "schema_version": 1,
        "artifact_type": "empty_reference_rgb_production_inputs",
        "data_origin": "twin",
        "input_fields": ["rgb"],
        "threshold_source": THRESHOLD_SOURCE,
        "split_version": "whole-run-varied-v3",
        "accepted_evidence": {
            "recovered_result_path": os.fspath(recovery_path),
            "recovered_result_sha256": ACCEPTED_RECOVERY_SHA256,
            "independent_review_path": os.fspath(review_path),
            "independent_review_sha256": ACCEPTED_REVIEW_SHA256,
        },
        "source_bindings": {
            "image_locators_path": os.fspath(locator_path),
            "image_locators_sha256": CORRECTED_V3_LOCATORS_SHA256,
            "downstream_validation_receipt_path": os.fspath(validation_path),
            "downstream_validation_receipt_sha256": CORRECTED_V3_VALIDATION_SHA256,
            "request_bundle_path": os.fspath(bundle_path),
            "request_bundle_sha256": CORRECTED_V3_REQUEST_BUNDLE_SHA256,
        },
        "records": records,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_new(destination, _canonical_bytes(artifact))
    return artifact


__all__ = [
    "ACCEPTED_V3_SAFE_MANIFEST_SHA256",
    "ACCEPTED_RECOVERY_SHA256",
    "ACCEPTED_REVIEW_SHA256",
    "PREDICTOR_CODE_COMMIT",
    "PREDICTOR_CONFIG",
    "PREDICTOR_CONFIG_SHA256",
    "PREDICTOR_SOURCE_SHA256",
    "PredictedMaskMaterializationError",
    "materialize_predictions",
    "prepare_v3_safe_manifest",
]

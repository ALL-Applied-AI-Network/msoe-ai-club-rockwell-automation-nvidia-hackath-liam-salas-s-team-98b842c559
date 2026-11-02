"""Training utilities for the synthetic FillCam four-class segmenter.

Torch stays behind function boundaries so contributor-side contract tests can run
without installing the Brev GPU environment locally. Dataset discovery deliberately
skips test chunks before opening a state file or manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageFilter


LABEL_IDS = frozenset((0, 1, 2, 3, 255))
SCORED_LABEL_IDS = (0, 1, 2, 3)
IGNORE_LABEL = 255
CROP_XYXY = (256, 48, 368, 464)
STEP6_EXPECTED_ROLE_COUNTS = {"train": 3000, "val": 300}
REVISION2_COMBINED_EXPECTED_ROLE_COUNTS = {"train": 3480, "val": 360}
# Backwards-compatible default for the committed Step 6 batch and legacy callers.
EXPECTED_ROLE_COUNTS = STEP6_EXPECTED_ROLE_COUNTS
LIQUID_RECOLOUR_PROBABILITY = 0.5
LIQUID_REFERENCE_HUE_FAMILIES = {
    "red_brown": (0.00, 0.06),
    "olive": (0.18, 0.22),
    "green_teal": (0.33, 0.52),
    "blue": (0.55, 0.59),
}
LIQUID_SATURATION_RANGE = (0.30, 0.85)
LIQUID_DARKEN_PROBABILITY = 0.10
LIQUID_VALUE_RANGE = (0.05, 0.09)
GLOBAL_BRIGHTNESS_RANGE = (0.70, 1.30)
GLOBAL_CONTRAST_RANGE = (0.75, 1.25)
GLOBAL_GAMMA_RANGE = (0.70, 1.40)
GLOBAL_SATURATION_RANGE = (0.50, 1.50)
GLOBAL_WHITE_BALANCE_RANGE = (0.85, 1.15)
JPEG_420_PROBABILITY = 0.75
JPEG_QUALITY_RANGE = (45, 95)
GAUSSIAN_BLUR_PROBABILITY = 0.5
GAUSSIAN_BLUR_RADIUS_RANGE = (0.1, 1.2)
GAUSSIAN_NOISE_PROBABILITY = 0.6
GAUSSIAN_NOISE_SIGMA_RANGE = (0.5, 8.0)
RGB8_HSV_QUANTIZATION_NOTE = (
    "HSV values are requested in 8-bit HSV and returned as 8-bit RGB; measuring the "
    "RGB result in HSV can shift hue, saturation, value or requested range boundaries "
    "through integer quantization, especially for very dark or nearly neutral pixels."
)
CHUNK_PATTERN = re.compile(
    r"^(?P<index>\d{2})-(?P<role>train|val|test)-(?P<start>\d{7})-(?P<end>\d{7})$"
)


@dataclass(frozen=True)
class FillLevelSample:
    """One approved synthetic RGB/mask pair."""

    role: str
    seed: int
    chunk: str
    frame_index: int
    rgb_path: Path
    mask_path: Path
    rgb_sha256: str
    mask_sha256: str

    def as_portable_dict(self):
        value = asdict(self)
        value["rgb_path"] = self.rgb_path.as_posix()
        value["mask_path"] = self.mask_path.as_posix()
        return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path, description):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _normalized_role_counts(value, description):
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object with train and val counts")
    result = {}
    for role in ("train", "val"):
        count = value.get(role)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{description} {role} count must be a positive integer")
        result[role] = int(count)
    return result


def _expected_role_count_contract(summary, configured_counts):
    declared = None
    batch_contract = summary.get("batch_contract")
    if batch_contract is not None:
        if not isinstance(batch_contract, dict):
            raise ValueError("batch summary batch_contract must be an object")
        if "expected_role_counts" in batch_contract:
            declared = _normalized_role_counts(
                batch_contract["expected_role_counts"],
                "batch-summary expected role",
            )
    configured = (
        _normalized_role_counts(configured_counts, "configured expected role")
        if configured_counts is not None
        else None
    )
    if configured is not None and declared is not None and configured != declared:
        raise ValueError(
            f"configured expected role-count contract {configured} does not match "
            f"batch-summary contract {declared}"
        )
    if configured is not None:
        source = (
            "caller_and_batch_summary.batch_contract"
            if declared is not None
            else "caller"
        )
        return configured, source
    if declared is not None:
        return declared, "batch_summary.batch_contract"
    return dict(STEP6_EXPECTED_ROLE_COUNTS), "legacy_step6_default"


def validate_exception_review(
    batch_summary_path,
    review_path,
    *,
    expected_role_counts=None,
):
    """Validate review evidence against explicit, or legacy Step 6, population counts."""
    summary_path = Path(batch_summary_path)
    review_path = Path(review_path)
    summary = _read_json(summary_path, "batch summary")
    review = _read_json(review_path, "exception review")
    expected_counts, role_count_source = _expected_role_count_contract(
        summary,
        expected_role_counts,
    )
    if not summary.get("integrity", {}).get("passed"):
        raise ValueError("batch summary integrity must pass before training")
    if review.get("status") != "complete":
        raise ValueError("exception review status must be complete before training")
    observed_summary_hash = file_sha256(summary_path)
    if review.get("batch_summary_sha256") != observed_summary_hash:
        raise ValueError("exception review batch-summary hash does not match")

    expected = {
        int(item["seed"]): str(item["role"])
        for item in summary.get("old_rule_review", {}).get("exceptions", [])
    }
    reviewed_items = review.get("exceptions")
    if not isinstance(reviewed_items, list):
        raise ValueError("exception review exceptions must be a list")
    reviewed = {}
    excluded = set()
    for item in reviewed_items:
        if not isinstance(item, dict):
            raise ValueError("each reviewed exception must be an object")
        seed = int(item.get("seed"))
        if seed in reviewed:
            raise ValueError(f"exception seed {seed} is reviewed more than once")
        role = str(item.get("role"))
        disposition = item.get("disposition")
        if disposition not in ("accepted_legitimate_occlusion", "excluded"):
            raise ValueError(f"exception seed {seed} has invalid disposition")
        explanation = item.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError(f"exception seed {seed} needs a non-empty explanation")
        overlay_hash = item.get("overlay_sha256")
        if not isinstance(overlay_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", overlay_hash
        ):
            raise ValueError(f"exception seed {seed} needs an overlay SHA-256")
        reviewed[seed] = role
        if disposition == "excluded":
            excluded.add(seed)
    if reviewed != expected:
        raise ValueError(
            f"exception review seeds/roles do not match inventory: "
            f"expected {expected}, got {reviewed}"
        )
    excluded_train_val = {seed for seed in excluded if reviewed[seed] in ("train", "val")}
    excluded_roles = {
        role: sum(1 for seed in excluded_train_val if reviewed[seed] == role)
        for role in ("train", "val")
    }
    expected_eligible = {
        role: expected_counts[role] - excluded_roles[role]
        for role in ("train", "val")
    }
    if review.get("excluded_train_validation_seeds") != sorted(excluded_train_val):
        raise ValueError(
            "exception review excluded_train_validation_seeds does not match dispositions"
        )
    eligible_counts = review.get("eligible_counts")
    if eligible_counts != expected_eligible:
        raise ValueError(
            f"exception review eligible_counts must be {expected_eligible}; "
            f"got {eligible_counts}"
        )
    approved_chunks = {}
    for chunk in summary.get("chunks", []):
        if chunk.get("role") not in ("train", "val"):
            continue
        name = str(chunk.get("chunk"))
        if name in approved_chunks:
            raise ValueError(f"batch summary repeats chunk {name}")
        manifest_hash = chunk.get("manifest_sha256")
        if not isinstance(manifest_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", manifest_hash
        ):
            raise ValueError(f"batch summary chunk {name} has invalid manifest hash")
        approved_chunks[name] = {
            "role": str(chunk["role"]),
            "attempt": str(chunk["attempt"]),
            "manifest_sha256": manifest_hash,
            "seed_start": int(chunk["seed_start"]),
            "seed_end": int(chunk["seed_end"]),
        }
    if not approved_chunks:
        raise ValueError("batch summary has no approved train/validation chunks")
    return {
        "batch_summary_sha256": observed_summary_hash,
        "review_sha256": file_sha256(review_path),
        "reviewed": len(reviewed),
        "excluded_seeds": sorted(excluded_train_val),
        "excluded_roles": excluded_roles,
        "expected_role_counts": expected_counts,
        "role_count_source": role_count_source,
        "eligible_counts": expected_eligible,
        "approved_chunks": approved_chunks,
    }


def dataset_population_contract(audit):
    """Return the count contract recorded in training config and run evidence."""
    return {
        "expected_role_counts": dict(audit["expected_role_counts"]),
        "eligible_role_counts": dict(audit["eligible_counts"]),
        "role_count_source": str(audit["role_count_source"]),
    }


def _safe_output_path(output_dir, relative_path, description):
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe {description} path: {relative_path}")
    output_resolved = Path(output_dir).resolve()
    resolved = (output_resolved / relative).resolve()
    if resolved != output_resolved and output_resolved not in resolved.parents:
        raise ValueError(f"unsafe {description} path: {relative_path}")
    return resolved


def _complete_attempt(chunk_dir):
    attempts = []
    for path in sorted(Path(chunk_dir).iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        state_path = path / "state.txt"
        try:
            state = state_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if state == "COMPLETE":
            attempts.append(path)
    if len(attempts) != 1:
        raise ValueError(
            f"chunk {Path(chunk_dir).name} needs exactly one COMPLETE attempt; "
            f"found {len(attempts)}"
        )
    return attempts[0]


def _manifest_samples(chunk_dir, role, excluded_seeds, expected_chunk=None):
    attempt = _complete_attempt(chunk_dir)
    if expected_chunk is not None and attempt.name != expected_chunk["attempt"]:
        raise ValueError(
            f"chunk {Path(chunk_dir).name} COMPLETE attempt is {attempt.name}; "
            f"approved attempt is {expected_chunk['attempt']}"
        )
    output_dir = attempt / "output"
    manifest_path = output_dir / "manifest.jsonl"
    try:
        manifest_bytes = manifest_path.read_bytes()
        lines = manifest_bytes.decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"could not read manifest {manifest_path}: {error}") from error
    if (
        expected_chunk is not None
        and hashlib.sha256(manifest_bytes).hexdigest()
        != expected_chunk["manifest_sha256"]
    ):
        raise ValueError(f"chunk {Path(chunk_dir).name} manifest hash is not approved")
    samples = []
    for line_number, raw_line in enumerate(lines, 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid manifest JSON {manifest_path}:{line_number}: {error.msg}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"manifest record {manifest_path}:{line_number} is not an object")
        if record.get("role") != role:
            raise ValueError(
                f"manifest role mismatch in {manifest_path}:{line_number}: "
                f"expected {role}, got {record.get('role')}"
            )
        seed = int(record["seed"])
        if expected_chunk is not None and not (
            expected_chunk["seed_start"] <= seed <= expected_chunk["seed_end"]
        ):
            raise ValueError(
                f"manifest seed {seed} is outside approved chunk range "
                f"{expected_chunk['seed_start']}-{expected_chunk['seed_end']}"
            )
        if seed in excluded_seeds:
            continue
        outputs = record.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError(f"manifest record seed {seed} has no outputs object")
        try:
            rgb_relative = outputs["rgb"]["path"]
            mask_relative = outputs["mask"]["path"]
            rgb_hash = outputs["rgb"]["sha256"]
            mask_hash = outputs["mask"]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"manifest record seed {seed} has invalid output metadata") from error
        if not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (rgb_hash, mask_hash)
        ):
            raise ValueError(f"manifest record seed {seed} has invalid output hashes")
        samples.append(
            FillLevelSample(
                role=role,
                seed=seed,
                chunk=Path(chunk_dir).name,
                frame_index=int(record["frame_index"]),
                rgb_path=_safe_output_path(output_dir, rgb_relative, "RGB"),
                mask_path=_safe_output_path(output_dir, mask_relative, "mask"),
                rgb_sha256=rgb_hash,
                mask_sha256=mask_hash,
            )
        )
    return samples


def discover_train_val_samples(
    dataset_root,
    *,
    expected_train=EXPECTED_ROLE_COUNTS["train"],
    expected_val=EXPECTED_ROLE_COUNTS["val"],
    excluded_seeds=(),
    expected_chunks=None,
):
    """Discover only train/validation records; test chunks are not opened."""
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"dataset root is not a directory: {dataset_root}")
    excluded_seeds = frozenset(int(seed) for seed in excluded_seeds)
    by_role = {"train": [], "val": []}
    matching_chunk_count = 0
    observed_chunk_names = set()
    for chunk_dir in sorted(dataset_root.iterdir(), key=lambda item: item.name):
        match = CHUNK_PATTERN.fullmatch(chunk_dir.name)
        if not match or not chunk_dir.is_dir():
            continue
        role = match.group("role")
        if role == "test":
            continue
        observed_chunk_names.add(chunk_dir.name)
        matching_chunk_count += 1
        expected_chunk = None
        if expected_chunks is not None:
            expected_chunk = expected_chunks.get(chunk_dir.name)
            if expected_chunk is None:
                raise ValueError(f"train/validation chunk {chunk_dir.name} is not approved")
            if expected_chunk["role"] != role:
                raise ValueError(f"approved role mismatch for chunk {chunk_dir.name}")
        by_role[role].extend(
            _manifest_samples(chunk_dir, role, excluded_seeds, expected_chunk)
        )
    if matching_chunk_count == 0:
        raise ValueError(f"no train/validation chunks found below {dataset_root}")
    if expected_chunks is not None and observed_chunk_names != set(expected_chunks):
        raise ValueError(
            "train/validation chunk set does not match approved batch summary: "
            f"missing={sorted(set(expected_chunks) - observed_chunk_names)}, "
            f"extra={sorted(observed_chunk_names - set(expected_chunks))}"
        )
    expected = {"train": int(expected_train), "val": int(expected_val)}
    observed = {role: len(samples) for role, samples in by_role.items()}
    if observed != expected:
        raise ValueError(f"train/validation counts must be {expected}; got {observed}")
    all_seeds = [sample.seed for samples in by_role.values() for sample in samples]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("duplicate seed across train/validation samples")
    for samples in by_role.values():
        samples.sort(key=lambda sample: (sample.seed, sample.frame_index))
    return by_role


def verify_sample_files(samples):
    """Hash every selected RGB/mask once before a training run starts."""
    digest = hashlib.sha256()
    sample_count = 0
    file_count = 0
    for sample in samples:
        actual_rgb = file_sha256(sample.rgb_path)
        actual_mask = file_sha256(sample.mask_path)
        if actual_rgb != sample.rgb_sha256:
            raise ValueError(f"seed {sample.seed} RGB hash does not match approved manifest")
        if actual_mask != sample.mask_sha256:
            raise ValueError(f"seed {sample.seed} mask hash does not match approved manifest")
        digest.update(
            (
                f"{sample.role}\t{sample.seed}\t{actual_rgb}\t{actual_mask}\n"
            ).encode("utf-8")
        )
        sample_count += 1
        file_count += 2
    if sample_count == 0:
        raise ValueError("no selected sample files to verify")
    return {
        "samples_checked": sample_count,
        "files_checked": file_count,
        "ordered_hash_ledger_sha256": digest.hexdigest(),
    }


def validate_mask_array(mask):
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D; got shape {mask.shape}")
    seen = {int(value) for value in np.unique(mask)}
    unexpected = seen - LABEL_IDS
    if unexpected:
        raise ValueError(f"mask has unexpected class ids: {sorted(unexpected)}")
    if not (seen & set(SCORED_LABEL_IDS)):
        raise ValueError("mask has no scored pixels")
    return mask


def crop_rgb_mask(rgb, mask, crop_xyxy=CROP_XYXY):
    rgb = np.asarray(rgb)
    mask = validate_mask_array(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must be HxWx3; got shape {rgb.shape}")
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"RGB/mask shape mismatch: {rgb.shape[:2]} vs {mask.shape}")
    x0, y0, x1, y1 = (int(value) for value in crop_xyxy)
    if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
        raise ValueError(f"crop {crop_xyxy} is outside image shape {rgb.shape[:2]}")
    return np.ascontiguousarray(rgb[y0:y1, x0:x1]), np.ascontiguousarray(
        mask[y0:y1, x0:x1]
    )


def recolour_liquid_pixels(
    rgb,
    mask,
    rng,
    *,
    probability=LIQUID_RECOLOUR_PROBABILITY,
    hue_families=LIQUID_REFERENCE_HUE_FAMILIES,
    saturation_range=LIQUID_SATURATION_RANGE,
):
    """Set liquid HSV8 hue/S targets while carrying source V into RGB8 conversion."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = validate_mask_array(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != mask.shape:
        raise ValueError("RGB and mask must have matching HxW geometry")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("liquid-recolour probability must be between zero and one")
    if not isinstance(hue_families, dict) or not hue_families:
        raise ValueError("liquid-recolour hue families must be a non-empty mapping")
    normalized_families = []
    for name, bounds in hue_families.items():
        if not isinstance(name, str) or not name:
            raise ValueError("liquid-recolour hue-family names must be non-empty strings")
        low, high = (float(value) for value in bounds)
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                "liquid-recolour hue-family ranges must satisfy 0 <= low <= high <= 1"
            )
        normalized_families.append((name, low, high))
    saturation_low, saturation_high = (
        float(value) for value in saturation_range
    )
    if not 0.0 <= saturation_low <= saturation_high <= 1.0:
        raise ValueError(
            "liquid-recolour saturation range must satisfy 0 <= low <= high <= 1"
        )
    if probability == 0.0 or (
        probability < 1.0 and float(rng.random()) >= probability
    ):
        return np.ascontiguousarray(rgb.copy())
    liquid = mask == 1
    if not np.any(liquid):
        return np.ascontiguousarray(rgb.copy())

    _family_name, hue_low, hue_high = normalized_families[
        int(rng.integers(0, len(normalized_families)))
    ]
    hue = float(rng.uniform(hue_low, hue_high))
    saturation = float(rng.uniform(saturation_low, saturation_high))
    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV")).copy()
    hsv[..., 0][liquid] = int(round(hue * 255.0)) % 256
    hsv[..., 1][liquid] = int(round(saturation * 255.0))
    recoloured = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))
    result = rgb.copy()
    result[liquid] = recoloured[liquid]
    return np.ascontiguousarray(result)


def darken_liquid_pixels(
    rgb,
    mask,
    rng,
    *,
    probability=LIQUID_DARKEN_PROBABILITY,
    value_range=LIQUID_VALUE_RANGE,
):
    """Scale liquid HSV8 V texture; H/S are source targets before RGB8 conversion."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    mask = validate_mask_array(mask)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != mask.shape:
        raise ValueError("RGB and mask must have matching HxW geometry")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("liquid-darkening probability must be between zero and one")
    low, high = (float(value) for value in value_range)
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("liquid-darkening value range must satisfy 0 <= low < high <= 1")
    if probability == 0.0 or (
        probability < 1.0 and float(rng.random()) >= probability
    ):
        return np.ascontiguousarray(rgb.copy())
    liquid = mask == 1
    if not np.any(liquid):
        return np.ascontiguousarray(rgb.copy())

    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV")).copy()
    original_value = hsv[..., 2][liquid].astype(np.float32) / 255.0
    target_mean = float(rng.uniform(low, high))
    if float(original_value.mean()) > 0.0:
        recoloured_value = np.clip(
            original_value * (target_mean / float(original_value.mean())),
            0.0,
            1.0,
        )
    else:
        recoloured_value = np.full_like(original_value, target_mean)
    hsv[..., 2][liquid] = np.rint(recoloured_value * 255.0).astype(np.uint8)
    recoloured = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))
    result = rgb.copy()
    result[liquid] = recoloured[liquid]
    return np.ascontiguousarray(result)


def global_photometric_jitter(
    rgb,
    rng,
    *,
    brightness_range=GLOBAL_BRIGHTNESS_RANGE,
    contrast_range=GLOBAL_CONTRAST_RANGE,
    gamma_range=GLOBAL_GAMMA_RANGE,
    saturation_range=GLOBAL_SATURATION_RANGE,
    white_balance_range=GLOBAL_WHITE_BALANCE_RANGE,
):
    """Apply seeded full-frame brightness, colour and white-balance jitter."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must be HxWx3; got shape {rgb.shape}")

    def factor(bounds, description):
        low, high = (float(value) for value in bounds)
        if not 0.0 < low <= high:
            raise ValueError(f"{description} range must contain positive ordered values")
        return float(rng.uniform(low, high))

    result = rgb.astype(np.float32) / 255.0
    result *= factor(brightness_range, "brightness")
    contrast = factor(contrast_range, "contrast")
    midpoint = float(result.mean())
    result = (result - midpoint) * contrast + midpoint
    result = np.clip(result, 0.0, 1.0)
    result = np.power(result, factor(gamma_range, "gamma"))
    saturation = factor(saturation_range, "saturation")
    gray = np.sum(
        result * np.array((0.2126, 0.7152, 0.0722), dtype=np.float32),
        axis=2,
        keepdims=True,
    )
    result = gray + saturation * (result - gray)
    white_low, white_high = (float(value) for value in white_balance_range)
    if not 0.0 < white_low <= white_high:
        raise ValueError("white-balance range must contain positive ordered values")
    result *= rng.uniform(white_low, white_high, size=(1, 1, 3)).astype(np.float32)
    return np.ascontiguousarray(
        np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
    )


def training_augmentation_config():
    """Return the JSON-safe, frozen Revision 2 train-only augmentation contract."""
    return {
        "mask_guided_liquid_recolour": {
            "mask_class_id": 1,
            "probability": LIQUID_RECOLOUR_PROBABILITY,
            "hue_family_uniform": {
                name: list(bounds)
                for name, bounds in LIQUID_REFERENCE_HUE_FAMILIES.items()
            },
            "saturation_uniform": list(LIQUID_SATURATION_RANGE),
            "hue_and_saturation_semantics": "requested_pre_rgb8_targets",
            "value": "carry_source_per_pixel_into_rgb8_conversion",
        },
        "mask_guided_liquid_darkening": {
            "mask_class_id": 1,
            "probability": LIQUID_DARKEN_PROBABILITY,
            "value_uniform": list(LIQUID_VALUE_RANGE),
            "value_statistic": "requested_liquid_pixel_mean_before_rgb8_conversion",
            "hue": "carry_source_target_into_rgb8_conversion",
            "saturation": "carry_source_target_into_rgb8_conversion",
            "texture": "requested_multiplicative_value_scaling",
        },
        "rgb8_hsv_quantization": {
            "requested_ranges_are_targets_not_measured_guarantees": True,
            "note": RGB8_HSV_QUANTIZATION_NOTE,
        },
        "mask_guided_liquid_transforms_apply_to": (
            "all synthetic training samples"
        ),
        "mask_guided_liquid_transform_draws_are_independent": True,
        "global_photometric_jitter": {
            "brightness": list(GLOBAL_BRIGHTNESS_RANGE),
            "contrast": list(GLOBAL_CONTRAST_RANGE),
            "gamma": list(GLOBAL_GAMMA_RANGE),
            "saturation": list(GLOBAL_SATURATION_RANGE),
            "white_balance": list(GLOBAL_WHITE_BALANCE_RANGE),
        },
        "jpeg_420": {
            "probability": JPEG_420_PROBABILITY,
            "quality_uniform_inclusive": list(JPEG_QUALITY_RANGE),
            "subsampling": "4:2:0",
        },
        "gaussian_blur": {
            "probability": GAUSSIAN_BLUR_PROBABILITY,
            "radius_uniform": list(GAUSSIAN_BLUR_RADIUS_RANGE),
        },
        "gaussian_noise": {
            "probability": GAUSSIAN_NOISE_PROBABILITY,
            "sigma_uniform": list(GAUSSIAN_NOISE_SIGMA_RANGE),
        },
        "geometry_changes": False,
        "validation_augmentation": False,
    }


def augment_rgb(rgb, rng, mask=None):
    """Train-only photometric augmentation; mask geometry is never touched."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if mask is not None:
        rgb = recolour_liquid_pixels(rgb, mask, rng)
        rgb = darken_liquid_pixels(rgb, mask, rng)
    rgb = global_photometric_jitter(rgb, rng)
    image = Image.fromarray(rgb, mode="RGB")
    if float(rng.random()) < JPEG_420_PROBABILITY:
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1)),
            subsampling=2,
            optimize=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
    if float(rng.random()) < GAUSSIAN_BLUR_PROBABILITY:
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=float(rng.uniform(*GAUSSIAN_BLUR_RADIUS_RANGE))
            )
        )
    result = np.asarray(image, dtype=np.int16)
    if float(rng.random()) < GAUSSIAN_NOISE_PROBABILITY:
        sigma = float(rng.uniform(*GAUSSIAN_NOISE_SIGMA_RANGE))
        result = result + rng.normal(0.0, sigma, size=result.shape).round().astype(np.int16)
    return np.ascontiguousarray(np.clip(result, 0, 255).astype(np.uint8))


def mask_class_counts(mask_paths, crop_xyxy=CROP_XYXY):
    """Count cropped scored pixels from training masks only."""
    counts = np.zeros(len(SCORED_LABEL_IDS), dtype=np.int64)
    for path in mask_paths:
        with Image.open(path) as image:
            mask = np.asarray(image)
        validate_mask_array(mask)
        x0, y0, x1, y1 = crop_xyxy
        cropped = mask[y0:y1, x0:x1]
        for class_id in SCORED_LABEL_IDS:
            counts[class_id] += int(np.count_nonzero(cropped == class_id))
    if np.any(counts <= 0):
        raise ValueError(f"every scored class needs training pixels; got {counts.tolist()}")
    return counts


def class_weights_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (len(SCORED_LABEL_IDS),) or np.any(counts <= 0):
        raise ValueError("class counts must contain four positive values")
    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights /= weights.mean()
    return weights.astype(np.float32)


def confusion_matrix(prediction, target, num_classes=4, ignore_index=IGNORE_LABEL):
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    valid = target != ignore_index
    valid &= (target >= 0) & (target < num_classes)
    valid &= (prediction >= 0) & (prediction < num_classes)
    encoded = target[valid].astype(np.int64) * num_classes + prediction[valid].astype(
        np.int64
    )
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def iou_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion matrix must be square")
    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    per_class = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan, dtype=np.float64),
        where=union > 0,
    )
    valid = np.isfinite(per_class)
    macro = float(np.mean(per_class[valid])) if np.any(valid) else math.nan
    return per_class, macro


def make_torch_dataset(samples, *, augment, seed, crop_xyxy=CROP_XYXY):
    """Build a Torch Dataset lazily after the caller has imported Torch."""
    import torch

    class FillLevelDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.samples = tuple(samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sample = self.samples[index]
            with Image.open(sample.rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(sample.mask_path) as image:
                mask = np.asarray(image)
            rgb, mask = crop_rgb_mask(rgb, mask, crop_xyxy)
            if augment:
                worker = torch.utils.data.get_worker_info()
                worker_seed = worker.seed if worker is not None else int(seed)
                rng = np.random.default_rng(
                    (int(worker_seed) + int(sample.seed) * 1_000_003) % (2**63 - 1)
                )
                rgb = augment_rgb(rgb, rng, mask)
            image_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255.0)
            mask_tensor = torch.from_numpy(mask.astype(np.int64, copy=False))
            return image_tensor, mask_tensor, int(sample.seed)

    return FillLevelDataset()


def build_unet(num_classes=4, base_channels=16):
    """Construct the small four-level U-Net used by Step 7."""
    import torch

    nn = torch.nn

    class DoubleConv(nn.Module):
        def __init__(self, input_channels, output_channels):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, value):
            return self.layers(value)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = int(base_channels)
            self.enc1 = DoubleConv(3, b)
            self.enc2 = DoubleConv(b, b * 2)
            self.enc3 = DoubleConv(b * 2, b * 4)
            self.enc4 = DoubleConv(b * 4, b * 8)
            self.bottleneck = DoubleConv(b * 8, b * 16)
            self.pool = nn.MaxPool2d(2)
            self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2)
            self.dec4 = DoubleConv(b * 16, b * 8)
            self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
            self.dec3 = DoubleConv(b * 8, b * 4)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
            self.dec2 = DoubleConv(b * 4, b * 2)
            self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
            self.dec1 = DoubleConv(b * 2, b)
            self.head = nn.Conv2d(b, int(num_classes), 1)

        def forward(self, value):
            e1 = self.enc1(value)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            bottleneck = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat((self.up4(bottleneck), e4), dim=1))
            d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
            d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
            d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
            return self.head(d1)

    return UNet()


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    del worker_id
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

"""CPU contract tests for Step 7 synthetic-only U-Net training."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fluidmix import fill_level_training as TRAINING  # noqa: E402


SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "train_fill_level_unet",
    ROOT / "scripts" / "train_fill_level_unet.py",
)
TRAIN_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(TRAIN_SCRIPT)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_chunk(root, name, role, seeds, *, attempts=("attempt-01",)):
    chunk = root / name
    for attempt_name in attempts:
        attempt = chunk / attempt_name
        output = attempt / "output"
        output.mkdir(parents=True)
        (attempt / "state.txt").write_text("COMPLETE\n", encoding="utf-8")
        records = []
        for index, seed in enumerate(seeds):
            records.append(
                {
                    "role": role,
                    "seed": seed,
                    "frame_index": index,
                    "outputs": {
                        "rgb": {
                            "path": f"rgb/{index:05d}.png",
                            "sha256": "a" * 64,
                        },
                        "mask": {
                            "path": f"mask/{index:05d}.png",
                            "sha256": "b" * 64,
                        },
                    },
                }
            )
        (output / "manifest.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
    return chunk


def _review_files(
    tmp_path,
    *,
    status="complete",
    disposition="accepted_legitimate_occlusion",
    expected_counts=None,
    declare_counts=False,
    expected_test=500,
):
    expected_counts = dict(expected_counts or {"train": 3000, "val": 300})
    summary = {
        "integrity": {"passed": True},
        "old_rule_review": {"exceptions": [{"seed": 100327, "role": "train"}]},
        "chunks": [
            {
                "chunk": "02-train-0100000-0100199",
                "role": "train",
                "attempt": "attempt-01",
                "manifest_sha256": "b" * 64,
                "seed_start": 100000,
                "seed_end": 100199,
            }
        ],
    }
    if declare_counts:
        summary["batch_contract"] = {
            "expected_role_counts": {**expected_counts, "test": int(expected_test)}
        }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    excluded = [100327] if disposition == "excluded" else []
    review = {
        "status": status,
        "batch_summary_sha256": _sha256(summary_path),
        "eligible_counts": {
            "train": expected_counts["train"] - len(excluded),
            "val": expected_counts["val"],
        },
        "excluded_train_validation_seeds": excluded,
        "exceptions": [
            {
                "seed": 100327,
                "role": "train",
                "disposition": disposition,
                "explanation": "fully hidden below the base edge",
                "overlay_sha256": "a" * 64,
            }
        ],
    }
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return summary_path, review_path


def test_completed_exception_review_is_required_and_hashed(tmp_path):
    summary, review = _review_files(tmp_path)
    result = TRAINING.validate_exception_review(summary, review)
    assert result["eligible_counts"] == {"train": 3000, "val": 300}
    assert result["expected_role_counts"] == {"train": 3000, "val": 300}
    assert result["role_count_source"] == "legacy_step6_default"
    assert result["excluded_seeds"] == []
    value = json.loads(review.read_text())
    value["status"] = "pending"
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="status must be complete"):
        TRAINING.validate_exception_review(summary, review)


def test_cli_gate_fails_before_dataset_discovery_or_output_creation(tmp_path):
    summary, review = _review_files(tmp_path, status="pending")
    args = type(
        "Args",
        (),
        {
            "batch_summary": summary,
            "exception_review": review,
            "dataset_root": tmp_path / "must-not-be-opened",
            "output_root": tmp_path / "must-not-be-created",
        },
    )()
    with pytest.raises(ValueError, match="status must be complete"):
        TRAIN_SCRIPT.run(args)
    assert not args.output_root.exists()


def test_excluded_disposition_updates_declared_eligible_counts(tmp_path):
    summary, review = _review_files(tmp_path, disposition="excluded")
    result = TRAINING.validate_exception_review(summary, review)
    assert result["excluded_seeds"] == [100327]
    assert result["eligible_counts"] == {"train": 2999, "val": 300}


def test_combined_population_counts_come_from_explicit_batch_contract(tmp_path):
    combined = TRAINING.REVISION2_COMBINED_EXPECTED_ROLE_COUNTS
    assert combined == {"train": 3480, "val": 360}
    summary, review = _review_files(
        tmp_path,
        expected_counts=combined,
        declare_counts=True,
        expected_test=560,
    )

    result = TRAINING.validate_exception_review(summary, review)

    assert result["expected_role_counts"] == combined
    assert result["eligible_counts"] == combined
    assert result["role_count_source"] == "batch_summary.batch_contract"


def test_population_counts_can_be_configured_but_must_match_declared_contract(tmp_path):
    additional = {"train": 480, "val": 60}
    summary, review = _review_files(
        tmp_path,
        expected_counts=additional,
        declare_counts=True,
        expected_test=60,
    )
    result = TRAINING.validate_exception_review(
        summary,
        review,
        expected_role_counts=additional,
    )
    assert result["expected_role_counts"] == additional
    assert result["role_count_source"] == "caller_and_batch_summary.batch_contract"
    with pytest.raises(ValueError, match="does not match batch-summary contract"):
        TRAINING.validate_exception_review(
            summary,
            review,
            expected_role_counts={"train": 3000, "val": 300},
        )


def test_explicit_aggregate_population_contract_drives_bound_discovery(tmp_path):
    root = tmp_path / "combined-collected"
    chunks = (
        ("00-val-1200000-1200001", "val", [1200000, 1200001]),
        ("01-train-0300000-0300002", "train", [300000, 300001, 300002]),
    )
    approved = []
    for name, role, seeds in chunks:
        _write_chunk(root, name, role, seeds)
        manifest = root / name / "attempt-01" / "output" / "manifest.jsonl"
        approved.append(
            {
                "chunk": name,
                "role": role,
                "attempt": "attempt-01",
                "manifest_sha256": _sha256(manifest),
                "seed_start": seeds[0],
                "seed_end": seeds[-1],
            }
        )
    summary = {
        "integrity": {"passed": True},
        "batch_contract": {
            "expected_role_counts": {"train": 3, "val": 2, "test": 1}
        },
        "old_rule_review": {"exceptions": []},
        "chunks": approved,
    }
    summary_path = tmp_path / "combined-summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    review = {
        "status": "complete",
        "batch_summary_sha256": _sha256(summary_path),
        "eligible_counts": {"train": 3, "val": 2},
        "excluded_train_validation_seeds": [],
        "exceptions": [],
    }
    review_path = tmp_path / "combined-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    audit = TRAINING.validate_exception_review(summary_path, review_path)
    discovered = TRAINING.discover_train_val_samples(
        root,
        expected_train=audit["eligible_counts"]["train"],
        expected_val=audit["eligible_counts"]["val"],
        excluded_seeds=audit["excluded_seeds"],
        expected_chunks=audit["approved_chunks"],
    )

    assert [sample.seed for sample in discovered["train"]] == [300000, 300001, 300002]
    assert [sample.seed for sample in discovered["val"]] == [1200000, 1200001]


def test_exception_review_rejects_missing_inventory_item(tmp_path):
    summary, review = _review_files(tmp_path)
    value = json.loads(review.read_text())
    value["exceptions"] = []
    review.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="do not match inventory"):
        TRAINING.validate_exception_review(summary, review)


def test_discovery_skips_test_chunk_before_opening_attempt_or_manifest(tmp_path):
    root = tmp_path / "collected"
    _write_chunk(root, "00-val-1100000-1100000", "val", [1100000])
    _write_chunk(root, "02-train-0100000-0100001", "train", [100000, 100001])
    # An intentionally empty test chunk would fail immediately if discovery opened it.
    (root / "17-test-2100000-2100199").mkdir(parents=True)
    result = TRAINING.discover_train_val_samples(
        root,
        expected_train=2,
        expected_val=1,
    )
    assert [sample.seed for sample in result["train"]] == [100000, 100001]
    assert [sample.seed for sample in result["val"]] == [1100000]


def test_discovery_applies_exclusion_and_exact_post_exclusion_counts(tmp_path):
    root = tmp_path / "collected"
    _write_chunk(root, "00-val-1100000-1100000", "val", [1100000])
    _write_chunk(root, "02-train-0100000-0100001", "train", [100000, 100001])
    result = TRAINING.discover_train_val_samples(
        root,
        expected_train=1,
        expected_val=1,
        excluded_seeds=[100001],
    )
    assert [sample.seed for sample in result["train"]] == [100000]
    with pytest.raises(ValueError, match="counts must be"):
        TRAINING.discover_train_val_samples(
            root,
            expected_train=2,
            expected_val=1,
            excluded_seeds=[100001],
        )


def test_discovery_rejects_ambiguous_complete_attempts(tmp_path):
    root = tmp_path / "collected"
    _write_chunk(root, "00-val-1100000-1100000", "val", [1100000])
    _write_chunk(
        root,
        "02-train-0100000-0100000",
        "train",
        [100000],
        attempts=("attempt-01", "attempt-02"),
    )
    with pytest.raises(ValueError, match="exactly one COMPLETE"):
        TRAINING.discover_train_val_samples(root, expected_train=1, expected_val=1)


def test_discovery_binds_attempt_and_manifest_to_approved_summary(tmp_path):
    root = tmp_path / "collected"
    val_name = "00-val-1100000-1100000"
    train_name = "02-train-0100000-0100000"
    _write_chunk(root, val_name, "val", [1100000])
    _write_chunk(root, train_name, "train", [100000])
    expected = {}
    for name, role, seed in (
        (val_name, "val", 1100000),
        (train_name, "train", 100000),
    ):
        manifest = root / name / "attempt-01" / "output" / "manifest.jsonl"
        expected[name] = {
            "role": role,
            "attempt": "attempt-01",
            "manifest_sha256": _sha256(manifest),
            "seed_start": seed,
            "seed_end": seed,
        }
    TRAINING.discover_train_val_samples(
        root,
        expected_train=1,
        expected_val=1,
        expected_chunks=expected,
    )
    manifest = root / train_name / "attempt-01" / "output" / "manifest.jsonl"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="manifest hash is not approved"):
        TRAINING.discover_train_val_samples(
            root,
            expected_train=1,
            expected_val=1,
            expected_chunks=expected,
        )


def test_preflight_rejects_rgb_or_mask_tampering(tmp_path):
    rgb = tmp_path / "rgb.png"
    mask = tmp_path / "mask.png"
    rgb.write_bytes(b"approved-rgb")
    mask.write_bytes(b"approved-mask")
    sample = TRAINING.FillLevelSample(
        role="train",
        seed=100000,
        chunk="02-train-0100000-0100199",
        frame_index=0,
        rgb_path=rgb,
        mask_path=mask,
        rgb_sha256=_sha256(rgb),
        mask_sha256=_sha256(mask),
    )
    result = TRAINING.verify_sample_files([sample])
    assert result["samples_checked"] == 1
    assert result["files_checked"] == 2
    rgb.write_bytes(b"tampered-rgb")
    with pytest.raises(ValueError, match="RGB hash does not match"):
        TRAINING.verify_sample_files([sample])


def test_state_ledger_is_append_only(tmp_path):
    state = tmp_path / "state.jsonl"
    TRAIN_SCRIPT._append_state(state, "RUNNING")
    first = state.read_text()
    TRAIN_SCRIPT._append_state(state, "COMPLETE")
    lines = state.read_text().splitlines()
    assert len(lines) == 2
    assert state.read_text().startswith(first)
    assert [json.loads(line)["status"] for line in lines] == ["RUNNING", "COMPLETE"]


def test_crop_and_mask_contract():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.full((480, 640), 255, dtype=np.uint8)
    mask[48:464, 256:368] = 1
    cropped_rgb, cropped_mask = TRAINING.crop_rgb_mask(rgb, mask)
    assert cropped_rgb.shape == (416, 112, 3)
    assert cropped_mask.shape == (416, 112)
    assert set(np.unique(cropped_mask)) == {1}
    mask[100, 300] = 9
    with pytest.raises(ValueError, match="unexpected class ids"):
        TRAINING.crop_rgb_mask(rgb, mask)


def test_photometric_augmentation_preserves_dimensions_and_mask_geometry():
    rgb = np.tile(np.arange(112, dtype=np.uint8), (416, 1))
    rgb = np.stack((rgb, rgb, rgb), axis=2)
    mask = np.zeros((416, 112), dtype=np.uint8)
    mask[200:, :] = 1
    original_mask = mask.copy()
    augmented = TRAINING.augment_rgb(rgb, np.random.default_rng(7))
    assert augmented.shape == rgb.shape
    assert augmented.dtype == np.uint8
    assert np.array_equal(mask, original_mask)


def test_mask_guided_liquid_recolour_targets_hsv_and_changes_only_liquid_pixels():
    rgb = np.full((4, 5, 3), (90, 120, 150), dtype=np.uint8)
    rgb[2, :4] = np.array(
        ((48, 32, 16), (96, 64, 32), (144, 96, 48), (192, 128, 64)),
        dtype=np.uint8,
    )
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[2, :4] = 1

    first = TRAINING.recolour_liquid_pixels(
        rgb,
        mask,
        np.random.default_rng(1234),
        probability=1.0,
        hue_families={"fixed_olive": (0.20, 0.20)},
        saturation_range=(0.50, 0.50),
    )
    second = TRAINING.recolour_liquid_pixels(
        rgb,
        mask,
        np.random.default_rng(1234),
        probability=1.0,
        hue_families={"fixed_olive": (0.20, 0.20)},
        saturation_range=(0.50, 0.50),
    )

    assert np.array_equal(first, second)
    assert np.array_equal(first[mask != 1], rgb[mask != 1])
    original_hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"))
    recoloured_hsv = np.asarray(Image.fromarray(first, "RGB").convert("HSV"))
    value_error = np.abs(
        recoloured_hsv[..., 2][mask == 1].astype(np.int16)
        - original_hsv[..., 2][mask == 1].astype(np.int16)
    )
    assert int(value_error.max()) <= 1
    hue_error = np.abs(recoloured_hsv[..., 0][mask == 1].astype(np.int16) - 51)
    hue_error = np.minimum(hue_error, 256 - hue_error)
    assert int(hue_error.max()) <= 2
    saturation_error = np.abs(
        recoloured_hsv[..., 1][mask == 1].astype(np.int16) - 128
    )
    assert int(saturation_error.max()) <= 2
    unchanged = TRAINING.recolour_liquid_pixels(
        rgb,
        mask,
        np.random.default_rng(1234),
        probability=0.0,
    )
    assert np.array_equal(unchanged, rgb)


def test_mask_guided_liquid_darkening_targets_source_hs_and_changes_only_liquid_pixels():
    rgb = np.full((4, 5, 3), (90, 120, 150), dtype=np.uint8)
    rgb[2, :4] = np.array(
        ((48, 32, 16), (96, 64, 32), (144, 96, 48), (192, 128, 64)),
        dtype=np.uint8,
    )
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[2, :4] = 1

    first = TRAINING.darken_liquid_pixels(
        rgb,
        mask,
        np.random.default_rng(1234),
        probability=1.0,
        value_range=(0.05, 0.09),
    )
    second = TRAINING.darken_liquid_pixels(
        rgb,
        mask,
        np.random.default_rng(1234),
        probability=1.0,
        value_range=(0.05, 0.09),
    )

    assert np.array_equal(first, second)
    assert np.array_equal(first[mask != 1], rgb[mask != 1])
    original_hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV"))
    darkened_hsv = np.asarray(Image.fromarray(first, "RGB").convert("HSV"))
    hue_error = np.abs(
        darkened_hsv[..., 0][mask == 1].astype(np.int16)
        - original_hsv[..., 0][mask == 1].astype(np.int16)
    )
    hue_error = np.minimum(hue_error, 256 - hue_error)
    saturation_error = np.abs(
        darkened_hsv[..., 1][mask == 1].astype(np.int16)
        - original_hsv[..., 1][mask == 1].astype(np.int16)
    )
    assert int(hue_error.max()) <= 2
    assert int(saturation_error.max()) <= 2
    liquid_v = darkened_hsv[..., 2][mask == 1] / 255.0
    assert 0.05 <= float(liquid_v.mean()) <= 0.09
    assert np.all(np.diff(liquid_v) > 0)
    unchanged = TRAINING.darken_liquid_pixels(
        rgb,
        mask,
        np.random.default_rng(1234),
        probability=0.0,
    )
    assert np.array_equal(unchanged, rgb)


def test_rgb8_round_trip_can_move_measured_saturation_outside_requested_target():
    rgb = np.ones((1, 1, 3), dtype=np.uint8)
    mask = np.ones((1, 1), dtype=np.uint8)

    recoloured = TRAINING.recolour_liquid_pixels(
        rgb,
        mask,
        np.random.default_rng(7),
        probability=1.0,
        hue_families={"fixed_olive": (0.20, 0.20)},
        saturation_range=(0.30, 0.30),
    )
    measured_hsv = np.asarray(Image.fromarray(recoloured, "RGB").convert("HSV"))

    assert measured_hsv[0, 0, 2] == 1
    assert measured_hsv[0, 0, 1] == 0
    assert "integer quantization" in TRAINING.RGB8_HSV_QUANTIZATION_NOTE


def test_shared_training_augmentation_runs_liquid_transforms_before_global_effects(
    monkeypatch,
):
    rgb = np.full((4, 5, 3), 100, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[2, :4] = 1
    calls = []

    def recolour(image, observed_mask, rng):
        del rng
        calls.append("recolour")
        assert np.array_equal(observed_mask, mask)
        result = image.copy()
        result[observed_mask == 1, 0] = 110
        return result

    def darken(image, observed_mask, rng):
        del rng
        calls.append("darken")
        assert np.all(image[observed_mask == 1, 0] == 110)
        result = image.copy()
        result[observed_mask == 1, 1] = 90
        return result

    def jitter(image, rng):
        del rng
        calls.append("global_jitter")
        assert np.all(image[mask == 1, 1] == 90)
        return image.copy()

    monkeypatch.setattr(TRAINING, "recolour_liquid_pixels", recolour)
    monkeypatch.setattr(TRAINING, "darken_liquid_pixels", darken)
    monkeypatch.setattr(TRAINING, "global_photometric_jitter", jitter)
    monkeypatch.setattr(TRAINING, "JPEG_420_PROBABILITY", 0.0)
    monkeypatch.setattr(TRAINING, "GAUSSIAN_BLUR_PROBABILITY", 0.0)
    monkeypatch.setattr(TRAINING, "GAUSSIAN_NOISE_PROBABILITY", 0.0)

    augmented = TRAINING.augment_rgb(rgb, np.random.default_rng(55), mask)

    assert calls == ["recolour", "darken", "global_jitter"]
    assert np.all(augmented[mask == 1, 0] == 110)
    assert np.all(augmented[mask == 1, 1] == 90)
    assert np.array_equal(augmented[mask != 1], rgb[mask != 1])


def test_global_photometric_jitter_is_seeded_and_geometry_preserving():
    rows = np.linspace(20, 220, 12, dtype=np.uint8)[:, None]
    rgb = np.repeat(rows, 10, axis=1)
    rgb = np.stack((rgb, np.flipud(rgb), np.full_like(rgb, 110)), axis=2)

    first = TRAINING.global_photometric_jitter(rgb, np.random.default_rng(88))
    second = TRAINING.global_photometric_jitter(rgb, np.random.default_rng(88))
    different = TRAINING.global_photometric_jitter(rgb, np.random.default_rng(89))

    assert first.shape == rgb.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first, rgb)
    assert not np.array_equal(first, different)


def test_revision2_training_augmentation_contract_is_auditable():
    config = TRAINING.training_augmentation_config()
    assert config["mask_guided_liquid_recolour"] == {
        "mask_class_id": 1,
        "probability": 0.5,
        "hue_family_uniform": {
            "red_brown": [0.0, 0.06],
            "olive": [0.18, 0.22],
            "green_teal": [0.33, 0.52],
            "blue": [0.55, 0.59],
        },
        "saturation_uniform": [0.3, 0.85],
        "hue_and_saturation_semantics": "requested_pre_rgb8_targets",
        "value": "carry_source_per_pixel_into_rgb8_conversion",
    }
    assert config["mask_guided_liquid_darkening"] == {
        "mask_class_id": 1,
        "probability": 0.1,
        "value_uniform": [0.05, 0.09],
        "value_statistic": "requested_liquid_pixel_mean_before_rgb8_conversion",
        "hue": "carry_source_target_into_rgb8_conversion",
        "saturation": "carry_source_target_into_rgb8_conversion",
        "texture": "requested_multiplicative_value_scaling",
    }
    assert config["rgb8_hsv_quantization"] == {
        "requested_ranges_are_targets_not_measured_guarantees": True,
        "note": TRAINING.RGB8_HSV_QUANTIZATION_NOTE,
    }
    assert config["mask_guided_liquid_transforms_apply_to"] == (
        "all synthetic training samples"
    )
    assert config["mask_guided_liquid_transform_draws_are_independent"] is True
    assert config["global_photometric_jitter"]["brightness"] == [0.7, 1.3]
    assert config["global_photometric_jitter"]["contrast"] == [0.75, 1.25]
    assert config["global_photometric_jitter"]["gamma"] == [0.7, 1.4]
    assert config["global_photometric_jitter"]["saturation"] == [0.5, 1.5]
    assert config["global_photometric_jitter"]["white_balance"] == [0.85, 1.15]
    assert config["jpeg_420"]["probability"] == 0.75
    assert config["gaussian_blur"]["probability"] == 0.5
    assert config["gaussian_noise"]["probability"] == 0.6
    assert config["geometry_changes"] is False
    assert config["validation_augmentation"] is False


def test_revision1_cli_records_cosine_learning_rate_decay():
    args = TRAIN_SCRIPT._parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--batch-summary",
            "batch.json",
            "--exception-review",
            "review.json",
            "--output-root",
            "output",
            "--profile",
            "full",
        ]
    )
    profile = TRAIN_SCRIPT._profile_config(args)
    assert TRAIN_SCRIPT._scheduler_config(args, profile) == {
        "name": "cosine_annealing",
        "initial_learning_rate": 0.001,
        "minimum_learning_rate": 0.00001,
        "t_max_epochs": 30,
        "step_unit": "epoch",
    }
    args.min_learning_rate = args.learning_rate
    with pytest.raises(ValueError, match="minimum learning rate"):
        TRAIN_SCRIPT._scheduler_config(args, profile)


def test_training_config_and_report_population_contract_is_explicit():
    combined = TRAINING.dataset_population_contract(
        {
            "expected_role_counts": {"train": 3480, "val": 360},
            "eligible_counts": {"train": 3479, "val": 360},
            "role_count_source": "batch_summary.batch_contract",
        }
    )
    assert combined == {
        "expected_role_counts": {"train": 3480, "val": 360},
        "eligible_role_counts": {"train": 3479, "val": 360},
        "role_count_source": "batch_summary.batch_contract",
    }


def test_class_weights_use_four_positive_training_counts():
    weights = TRAINING.class_weights_from_counts([1000, 500, 10, 250])
    assert weights.shape == (4,)
    assert weights[2] > weights[1] > weights[0]
    assert weights.mean() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="four positive"):
        TRAINING.class_weights_from_counts([1, 2, 0, 4])


def test_ignore_aware_confusion_and_iou():
    target = np.array([[0, 1, 2], [3, 255, 1]], dtype=np.uint8)
    prediction = np.array([[0, 1, 1], [3, 0, 0]], dtype=np.uint8)
    confusion = TRAINING.confusion_matrix(prediction, target)
    assert confusion.sum() == 5
    assert confusion[0, 0] == 1
    assert confusion[1, 0] == 1
    assert confusion[2, 1] == 1
    per_class, macro = TRAINING.iou_from_confusion(confusion)
    assert per_class[3] == pytest.approx(1.0)
    assert 0.0 < macro < 1.0


def test_unet_output_shape_when_torch_is_available():
    torch = pytest.importorskip("torch")
    model = TRAINING.build_unet(base_channels=4)
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 416, 112)))
    assert tuple(output.shape) == (1, 4, 416, 112)
